//! N-dimensional pseudospectral memory-variable viscoacoustic solver.
//!
//! One canonical implementation covers 1-D, 2-D, and 3-D: a `(n,1,1)` grid is
//! 1-D, `(nx,ny,1)` is 2-D, and `(nx,ny,nz)` is full 3-D — the spectral
//! derivative along a singleton axis is identically zero. Lower-dimensional
//! grids therefore clear those derivative outputs without staging or FFT work,
//! and the solver owns **no storage for an inactive axis**: it retains
//! velocity, derivative, and wavenumber state only along axes that can carry
//! spatial variation (see [`ActiveAxes`]).

use kwavers_math::fft::{wavenumber_vec_positive_nyquist, Complex64, Fft3d};
use leto::Array3;
use std::sync::Arc;

mod axis;
mod boundary;
mod construction;
mod driven;
mod energy;
mod state;
mod time;

/// One relaxation arm with its precomputed exponential-integrator
/// coefficients (uniform coefficients for a homogeneous medium).
#[derive(Debug, Clone)]
struct Arm {
    /// `e^{-Δt/τₗ(x)}` (decay over one step).
    decay: Coeff,
    /// `−ΔMₗ(x)·τₗ(x)·(1 − e^{-Δt/τₗ})` — coefficient of `∇·v` in the σ update.
    gain: Coeff,
    /// `1/τₗ(x)` \[s⁻¹] — for the trapezoidal pressure contribution.
    inv_tau: Coeff,
}

/// A medium or integrator coefficient: one scalar when it is spatially
/// uniform, or the full per-voxel grid otherwise.
///
/// The representation specializes **once, at the operation boundary**: every
/// consumer branches on the variant per operation (per step, per arm) and
/// never inside a per-voxel loop — the scalar path hoists the value out of
/// the loop, the grid path streams the field. The scalar path performs the
/// same float operations in the same order as the grid path over a constant
/// field, so the two are bitwise comparable.
#[derive(Debug, Clone)]
enum Coeff {
    /// Spatially uniform: every voxel holds this value.
    Uniform(f64),
    /// Per-voxel field, grid-shaped and contiguous.
    Field(Array3<f64>),
}

impl Coeff {
    /// The scalar of a [`Coeff::Uniform`] coefficient.
    fn value(&self) -> f64 {
        match self {
            Coeff::Uniform(v) => *v,
            Coeff::Field(_) => unreachable!("invariant: scalar path reads uniform coefficients"),
        }
    }

    /// Whether the coefficient is spatially uniform.
    fn is_uniform(&self) -> bool {
        matches!(self, Coeff::Uniform(_))
    }

    /// The per-voxel slice of a [`Coeff::Field`] coefficient.
    fn slice(&self) -> &[f64] {
        match self {
            Coeff::Uniform(_) => unreachable!("invariant: grid path reads field coefficients"),
            Coeff::Field(a) => a
                .as_slice()
                .expect("invariant: viscoacoustic coefficient field is contiguous"),
        }
    }
}
/// Debug description of a coefficient's representation.
fn describe_coeff(c: &Coeff) -> String {
    match c {
        Coeff::Uniform(v) => format!("uniform({v})"),
        Coeff::Field(a) => format!("{:?}", a.shape()),
    }
}

/// Build an arm's exponential-integrator coefficient fields from per-voxel
/// relaxation strength `ΔMₗ(x)` and time `τₗ(x)`.
fn build_arm(delta_m: &Array3<f64>, tau: &Array3<f64>, dt: f64) -> Arm {
    let decay = tau.mapv(|t| (-dt / t).exp());
    let inv_tau = tau.mapv(|t| 1.0 / t);
    let mut gain = Array3::<f64>::zeros(delta_m.shape());
    leto_ops::zip_mut_with(
        gain.view_mut(),
        (&delta_m.view(), &tau.view(), &decay.view()),
        |g, (&dm, &t, &dc)| *g = -dm * t * (1.0 - dc),
    )
    .expect("invariant: build_arm operands share grid shape");
    Arm {
        decay: Coeff::Field(decay),
        gain: Coeff::Field(gain),
        inv_tau: Coeff::Field(inv_tau),
    }
}

/// Build an arm's exponential-integrator coefficients from scalar relaxation
/// strength `ΔMₗ` and time `τₗ` — the same formulas [`build_arm`] evaluates
/// per voxel, so a uniform field and the scalar produce bitwise-equal
/// coefficients.
fn build_uniform_arm(delta_m: f64, tau: f64, dt: f64) -> Arm {
    let decay = (-dt / tau).exp();
    Arm {
        decay: Coeff::Uniform(decay),
        gain: Coeff::Uniform(-delta_m * tau * (1.0 - decay)),
        inv_tau: Coeff::Uniform(1.0 / tau),
    }
}

/// Time-domain memory-variable viscoacoustic solver (1-D/2-D/3-D, pseudospectral).
///
/// Construct from a [`GeneralizedMaxwellModel`] (the relaxation spectrum) or from
/// raw moduli, then call [`Self::step`] to advance the velocity–pressure state.
#[derive(Clone)]
pub struct ViscoacousticMemorySolver {
    nx: usize,
    ny: usize,
    nz: usize,
    cell_volume: f64, // dx·dy·dz
    dt: f64,
    /// Which axes carry spatial variation and therefore own storage.
    axes: ActiveAxes,
    /// `1/ρ(x)` \[m³·kg⁻¹] for the velocity update — scalar when the medium
    /// is homogeneous.
    inv_rho: Coeff,
    /// Unrelaxed (instantaneous) modulus `M_U(x) = M_∞(x) + Σ ΔMₗ(x)` \\[Pa\]
    /// — scalar when the medium is homogeneous.
    m_u: Coeff,
    /// Equilibrium (relaxed) modulus `M_∞(x)` \\[Pa\] — the potential-energy
    /// norm; scalar when the medium is homogeneous.
    m_inf: Coeff,
    /// Maximum unrelaxed sound speed over the grid \[m·s⁻¹] — the CFL reference.
    max_unrelaxed_speed: f64,
    arms: Vec<Arm>,

    // Spectral derivative: apollo's batched, cache-tiled, parallel per-axis 3-D
    // FFT (forward_axis → ·ik → inverse_axis) reusing one complex scratch.
    fft: Arc<Fft3d>,
    kx: Vec<f64>,
    ky: Vec<f64>,
    kz: Vec<f64>,
    cbuf: Array3<Complex64>,

    // State.
    p: Array3<f64>,
    vx: Array3<f64>,
    vy: Array3<f64>,
    vz: Array3<f64>,
    sigma: Vec<Array3<f64>>, // one memory field per arm

    // Preallocated derivative/staging buffers, assigned per step in canonical
    // axis order: `gx` is the divergence output, `gy` the second staging slot
    // then the relaxation accumulator, `gz` the third slot (all-active only).
    gx: Array3<f64>,
    gy: Array3<f64>,
    gz: Array3<f64>,

    // Optional absorbing boundary: per-cell multiplicative decay `exp(-γ Δt)`
    // applied to `p` and `v` each step, with `γ` ramping up inside the boundary
    // layer and zero in the interior (`None` ⇒ periodic, non-absorbing).
    damping_decay: Option<Array3<f64>>,

    // Driven-simulation I/O.
    step_count: usize,
    /// Additive (soft) pressure sources: `(grid index, time signal)`.
    pressure_sources: Vec<((usize, usize, usize), Vec<f64>)>,
    /// Pressure-sensor grid indices and their recorded time traces.
    pressure_sensors: Vec<(usize, usize, usize)>,
    sensor_record: Vec<Vec<f64>>,
}

/// Which grid axes carry spatial variation and therefore own storage.
///
/// A singleton axis is periodic with a single sample: its spectral derivative
/// is identically zero for every field, so its velocity component stays zero
/// for all time and its derivative staging has no content. The solver omits
/// the state of an inactive axis entirely and substitutes the exact
/// `positive zero` identity at the operation boundary, so the canonical
/// 1-D/2-D/3-D grids `(n,1,1)`, `(nx,ny,1)`, `(nx,ny,nz)` keep their semantics
/// while the retained footprint drops by the inactive arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ActiveAxes {
    x: bool,
    y: bool,
    z: bool,
}

impl ActiveAxes {
    /// The active-axis set of a grid: an extent-1 axis is inactive.
    fn of(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            x: nx > 1,
            y: ny > 1,
            z: nz > 1,
        }
    }

    /// Number of active axes.
    fn count(self) -> usize {
        usize::from(self.x) + usize::from(self.y) + usize::from(self.z)
    }
}

impl std::fmt::Debug for ViscoacousticMemorySolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ViscoacousticMemorySolver")
            .field("nx", &self.nx)
            .field("ny", &self.ny)
            .field("nz", &self.nz)
            .field("dt", &self.dt)
            .field("cell_volume", &self.cell_volume)
            .field("inv_rho", &describe_coeff(&self.inv_rho))
            .field("m_u", &describe_coeff(&self.m_u))
            .field("m_inf", &describe_coeff(&self.m_inf))
            .field("max_unrelaxed_speed", &self.max_unrelaxed_speed)
            .field("arms", &(self.arms.len()))
            .field("fft", &"<fft-plan>")
            .field("axes", &self.axes)
            .field("kx", &self.kx.len())
            .field("ky", &self.ky.len())
            .field("kz", &self.kz.len())
            .field("cbuf", &self.cbuf.shape())
            .field("p", &self.p.shape())
            .field("vx", &self.vx.shape())
            .field("vy", &self.vy.shape())
            .field("vz", &self.vz.shape())
            .field("sigma", &self.sigma.len())
            .field("gx", &self.gx.shape())
            .field("gy", &self.gy.shape())
            .field("gz", &self.gz.shape())
            .field(
                "damping_decay",
                &self.damping_decay.as_ref().map(|v| v.shape()),
            )
            .field("step_count", &self.step_count)
            .field("pressure_sources", &self.pressure_sources.len())
            .field("pressure_sensors", &self.pressure_sensors)
            .field("sensor_record", &self.sensor_record.len())
            .finish()
    }
}

/// A constructor parameter is admissible only when it is finite and strictly
/// positive. `v > 0.0` alone is the wrong predicate for physical quantities:
/// every non-finite value compares false against `0.0`, so `NaN` and `+∞`
/// pass a bare positivity guard and reach retained coefficient, wavenumber,
/// and state construction. The finiteness half of this predicate is what the
/// finite-domain contract (ADR 129) adds; the positivity half is unchanged.
fn is_positive_finite(v: f64) -> bool {
    v.is_finite() && v > 0.0
}

/// The uniform Cartesian grid a viscoacoustic solve runs on.
///
/// Every constructor on [`ViscoacousticMemorySolver`] needs the same seven
/// numbers, and passing them positionally put them ahead of the material
/// parameters — so a caller writing `new(64, 64, 64, dx, dy, dz, dt, rho,
/// m_inf, &arms)` could not tell which spacing was which without counting.
/// The grid is one concept and the constructors now take it as one value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViscoacousticGrid {
    /// Cells along `x`.
    pub nx: usize,
    /// Cells along `y`.
    pub ny: usize,
    /// Cells along `z`.
    pub nz: usize,
    /// Cell size along `x`, `m`.
    pub dx: f64,
    /// Cell size along `y`, `m`.
    pub dy: f64,
    /// Cell size along `z`, `m`.
    pub dz: f64,
    /// Time step, `s`.
    pub dt: f64,
}

impl ViscoacousticGrid {
    /// Build a grid from its cell counts, spacings and time step.
    ///
    /// Seven numbers — at, not over, clippy's threshold — so this needs no
    /// suppression. A grid genuinely is seven scalars; splitting it further
    /// would be the same arity problem one level down.
    #[must_use]
    pub const fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64, dt: f64) -> Self {
        Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            dt,
        }
    }
}

/// FFT-order signed wavenumbers for a real first derivative.
///
/// On odd N the bins m and N−m are conjugates with opposite wavenumbers.
/// On even N the Nyquist bin is self-conjugate. Its imaginary multiplier
/// would leave the real signal subspace, so the derivative omits that bin.
fn fft_wavenumbers(n: usize, dx: f64) -> Vec<f64> {
    let mut modes = wavenumber_vec_positive_nyquist(n, dx);
    if n > 0 && n.is_multiple_of(2) {
        modes[n / 2] = 0.0;
    }
    modes
}
