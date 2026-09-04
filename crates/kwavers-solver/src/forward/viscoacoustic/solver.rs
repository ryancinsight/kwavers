//! N-dimensional pseudospectral memory-variable viscoacoustic solver.
//!
//! One canonical implementation covers 1-D, 2-D, and 3-D: a `(n,1,1)` grid is
//! 1-D, `(nx,ny,1)` is 2-D, and `(nx,ny,nz)` is full 3-D — the spectral
//! derivative along a singleton axis is identically zero. Lower-dimensional
//! grids therefore clear those derivative outputs without staging or FFT work,
//! and the solver owns **no storage for an inactive axis**: it retains
//! velocity, derivative, and wavenumber state only along axes that can carry
//! spatial variation (see [`ActiveAxes`]).

use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::fft::{get_fft_for_grid, Complex64, Fft3d};
use kwavers_medium::absorption::{fit_power_law_fields, FitBand};
use kwavers_medium::viscoelastic::GeneralizedMaxwellModel;
use leto::Array3 as LetoArray3;
use leto::Array3;
use std::sync::Arc;

mod axis;
mod driven;

/// One relaxation arm with its precomputed per-voxel exponential-integrator
/// coefficients (uniform fields for a homogeneous medium).
#[derive(Debug, Clone)]
struct Arm {
    /// `e^{-Δt/τₗ(x)}` (decay over one step).
    decay: Array3<f64>,
    /// `−ΔMₗ(x)·τₗ(x)·(1 − e^{-Δt/τₗ})` — coefficient of `∇·v` in the σ update.
    gain: Array3<f64>,
    /// `1/τₗ(x)` \[s⁻¹] — for the trapezoidal pressure contribution.
    inv_tau: Array3<f64>,
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
        decay,
        gain,
        inv_tau,
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
    /// Per-voxel `1/ρ(x)` \[m³·kg⁻¹] for the velocity update.
    inv_rho: Array3<f64>,
    /// Per-voxel unrelaxed (instantaneous) modulus `M_U(x) = M_∞(x) + Σ ΔMₗ(x)` \\[Pa\].
    m_u: Array3<f64>,
    /// Per-voxel equilibrium (relaxed) modulus `M_∞(x)` \\[Pa\] — potential-energy norm.
    m_inf: Array3<f64>,
    /// Maximum unrelaxed sound speed over the grid \[m·s⁻¹] — the CFL reference.
    max_unrelaxed_speed: f64,
    arms: Vec<Arm>,

    // Spectral derivative: apollo's batched, cache-tiled, parallel per-axis 3-D
    // FFT (forward_axis → ·ik → inverse_axis) reusing one complex scratch.
    fft: Arc<Fft3d>,
    kx: Vec<f64>,
    ky: Vec<f64>,
    kz: Vec<f64>,
    cbuf: LetoArray3<Complex64>,

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

    /// Every axis active — the full 3-D storage layout.
    fn all_active(nx: usize, ny: usize, nz: usize) -> bool {
        nx > 1 && ny > 1 && nz > 1
    }

    /// Number of active axes.
    fn count(self) -> usize {
        usize::from(self.x) + usize::from(self.y) + usize::from(self.z)
    }

    /// All singleton axes are inactive, so at least one axis is always active
    /// on a valid (non-degenerate but possibly singleton) grid.
    fn is_active(self, axis: usize) -> bool {
        match axis {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            _ => unreachable!("invariant: axis is 0, 1, or 2"),
        }
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
            .field("inv_rho", &self.inv_rho.shape())
            .field("m_u", &self.m_u.shape())
            .field("m_inf", &self.m_inf.shape())
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

impl ViscoacousticMemorySolver {
    /// Build from raw parameters: grid `(nx,ny,nz)` with spacings `(dx,dy,dz)`,
    /// time step `dt`, density `ρ`, equilibrium modulus `M_∞`, and relaxation
    /// arms `(ΔMₗ, τₗ)`. An empty arm list yields the lossless wave equation.
    /// # Errors
    /// - Any zero dimension, non-positive `dx`/`dy`/`dz`/`dt`/`ρ`/`M_∞`, or a
    ///   non-positive arm parameter.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        dt: f64,
        rho: f64,
        m_inf: f64,
        arms: &[(f64, f64)],
    ) -> KwaversResult<Self> {
        if nx == 0
            || ny == 0
            || nz == 0
            || dx <= 0.0
            || dy <= 0.0
            || dz <= 0.0
            || dt <= 0.0
            || rho <= 0.0
            || m_inf <= 0.0
        {
            return Err(KwaversError::InvalidInput(
                "viscoacoustic solver requires positive grid, spacings, dt, ρ, M_∞".to_owned(),
            ));
        }
        if arms.iter().any(|&(dm, tau)| dm <= 0.0 || tau <= 0.0) {
            return Err(KwaversError::InvalidInput(
                "relaxation arms require ΔM>0 and τ>0".to_owned(),
            ));
        }

        // Homogeneous medium: broadcast the scalar parameters to uniform fields
        // and delegate to the per-voxel assembler.
        let shape = (nx, ny, nz);
        let inv_rho = Array3::from_elem(shape, 1.0 / rho);
        let m_inf_field = Array3::from_elem(shape, m_inf);
        let arm_fields: Vec<Arm> = arms
            .iter()
            .map(|&(dm, tau)| {
                build_arm(
                    &Array3::from_elem(shape, dm),
                    &Array3::from_elem(shape, tau),
                    dt,
                )
            })
            .collect();
        Ok(Self::assemble(
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            dt,
            inv_rho,
            m_inf_field,
            arm_fields,
            ActiveAxes::of(nx, ny, nz),
        ))
    }

    /// Build a **heterogeneous** medium from per-voxel fields: density `ρ(x)`,
    /// equilibrium modulus `M_∞(x)`, and relaxation arms `(ΔMₗ(x), τₗ(x))` (all of
    /// grid shape). This lets a CT-derived tissue model (§4.5) drive the broadband
    /// solver with spatially-varying viscoacoustic properties.
    ///
    /// A zero-strength arm is inert (an exactly lossless voxel), so `ΔM` is
    /// only required non-negative; a heterogeneous medium may contain lossless
    /// regions alongside absorbing ones.
    ///
    /// # Errors
    /// - Any field shape ≠ `(nx,ny,nz)`, a non-positive `ρ`/`M_∞`/`τ`, a
    ///   negative `ΔM`, or
    ///   non-positive grid/spacing/`dt`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_heterogeneous(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        dt: f64,
        rho: &Array3<f64>,
        m_inf: &Array3<f64>,
        arms: &[(Array3<f64>, Array3<f64>)],
    ) -> KwaversResult<Self> {
        let shape = [nx, ny, nz];
        let ok_shape = |a: &Array3<f64>| a.shape() == shape;
        if nx == 0 || ny == 0 || nz == 0 || dx <= 0.0 || dy <= 0.0 || dz <= 0.0 || dt <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "viscoacoustic solver requires positive grid, spacings, dt".to_owned(),
            ));
        }
        if !ok_shape(rho)
            || !ok_shape(m_inf)
            || rho.iter().any(|&r| r <= 0.0)
            || m_inf.iter().any(|&m| m <= 0.0)
        {
            return Err(KwaversError::InvalidInput(
                "ρ and M_∞ fields must be grid-shaped and positive".to_owned(),
            ));
        }
        if arms.iter().any(|(dm, tau)| {
            !ok_shape(dm)
                || !ok_shape(tau)
                || dm.iter().any(|&v| v < 0.0)
                || tau.iter().any(|&v| v <= 0.0)
        }) {
            return Err(KwaversError::InvalidInput(
                "relaxation arm fields must be grid-shaped with ΔM≥0 and τ>0".to_owned(),
            ));
        }

        let inv_rho = rho.mapv(|r| 1.0 / r);
        let arm_fields: Vec<Arm> = arms
            .iter()
            .map(|(dm, tau)| build_arm(dm, tau, dt))
            .collect();
        Ok(Self::assemble(
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            dt,
            inv_rho,
            m_inf.clone(),
            arm_fields,
            ActiveAxes::of(nx, ny, nz),
        ))
    }

    /// Allocate state/scratch and assemble the solver from prepared per-voxel
    /// `inv_rho`, `m_inf`, and arm coefficient fields. `M_U = M_∞ + Σ ΔMₗ` is
    /// recovered from the arm gains; the CFL speed is the grid max of `√(M_U/ρ)`.
    ///
    /// `axes` is the storage mask. Production constructors derive it from the
    /// grid; the differential tests inject the all-active mask at a singleton
    /// grid to build the six-array reference the acceptance oracle compares
    /// against. The step code is identical under both layouts — an inactive
    /// axis is the exact positive-zero identity — so the two are bitwise
    /// comparable, and the injected reference is test-local, not a public
    /// constructor surface.
    #[allow(clippy::too_many_arguments)]
    fn assemble(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        dt: f64,
        inv_rho: Array3<f64>,
        m_inf: Array3<f64>,
        arms: Vec<Arm>,
        axes: ActiveAxes,
    ) -> Self {
        let shape = (nx, ny, nz);
        // M_U(x) = M_∞(x) + Σ ΔMₗ(x); recover ΔMₗ = −gain / (τ(1−decay)) = −gain·inv_tau/(1−decay).
        let mut m_u = m_inf.clone();
        for arm in &arms {
            leto_ops::zip_mut_with(
                m_u.view_mut(),
                (&arm.gain.view(), &arm.decay.view(), &arm.inv_tau.view()),
                |mu, (&gain, &decay, &inv_tau)| {
                    *mu += -gain * inv_tau / (1.0 - decay);
                },
            )
            .expect("invariant: viscoacoustic modulus fields share grid shape");
        }
        let max_unrelaxed_speed =
            leto_ops::zip_fold(&m_u.view(), &inv_rho.view(), 0.0_f64, |acc, &mu, &ir| {
                acc.max((mu * ir).sqrt())
            })
            .expect("invariant: modulus and inv_rho fields share grid shape");

        // Storage layout: velocity/wavenumber state only for active axes.
        // `gx` (divergence output) and `gy` (second staging slot, then the
        // relaxation accumulator) stay grid-shaped; `gz` is the third staging
        // slot and exists only in the all-active layout — the only step that
        // needs three simultaneous slots. An inactive axis owns no storage:
        // its velocity stays exactly zero for all time and its derivative is
        // the exact positive zero fill.
        let stage = |active: bool| {
            if active {
                Array3::zeros(shape)
            } else {
                Array3::zeros((0, 0, 0))
            }
        };
        let all_active = axes.x && axes.y && axes.z;

        Self {
            nx,
            ny,
            nz,
            cell_volume: dx * dy * dz,
            dt,
            axes,
            inv_rho,
            m_u,
            m_inf,
            max_unrelaxed_speed,
            sigma: vec![Array3::zeros(shape); arms.len()],
            arms,
            fft: get_fft_for_grid(nx, ny, nz),
            kx: if axes.x {
                fft_wavenumbers(nx, dx)
            } else {
                Vec::new()
            },
            ky: if axes.y {
                fft_wavenumbers(ny, dy)
            } else {
                Vec::new()
            },
            kz: if axes.z {
                fft_wavenumbers(nz, dz)
            } else {
                Vec::new()
            },
            cbuf: LetoArray3::from_elem([nx, ny, nz], Complex64::new(0.0, 0.0)),
            p: Array3::zeros(shape),
            vx: stage(axes.x),
            vy: stage(axes.y),
            vz: stage(axes.z),
            gx: Array3::zeros(shape),
            gy: Array3::zeros(shape),
            gz: stage(all_active),
            damping_decay: None,
            step_count: 0,
            pressure_sources: Vec::new(),
            pressure_sensors: Vec::new(),
            sensor_record: Vec::new(),
        }
    }

    /// Enable an **absorbing boundary layer** (sponge) of `thickness` cells on
    /// every face whose axis is long enough to hold it. Outgoing waves entering
    /// the layer are damped before they reach (and wrap around) the periodic
    /// boundary, suppressing artificial reflections.
    ///
    /// `gamma_max` \[s⁻¹] is the peak damping rate at the outermost cell; the
    /// rate ramps in as a quadratic profile `γ(d) = γ_max ((L-d)/L)²` over the
    /// layer depth `d ∈ [0, L)` (zero in the interior), summed across axes so
    /// corners damp in every direction. A smooth ramp keeps layer reflection low.
    /// Calling again rebuilds the profile; `thickness = 0` disables it.
    pub fn enable_absorbing_layer(&mut self, thickness: usize, gamma_max: f64) {
        if thickness == 0 || gamma_max <= 0.0 {
            self.damping_decay = None;
            return;
        }
        // Per-axis ramp: γ contribution at index `i` along an axis of extent `n`.
        let ramp = |i: usize, n: usize| -> f64 {
            if n <= 2 * thickness {
                return 0.0; // axis too short to host the layer (e.g. singleton)
            }
            let l = thickness as f64;
            let depth = if i < thickness {
                (thickness - i) as f64 // distance from the low boundary
            } else if i >= n - thickness {
                (i - (n - thickness) + 1) as f64 // distance from the high boundary
            } else {
                0.0
            };
            let frac = depth / l;
            gamma_max * frac * frac
        };

        let dt = self.dt;
        let decay = Array3::from_shape_fn((self.nx, self.ny, self.nz), |[i, j, k]| {
            let gamma = ramp(i, self.nx) + ramp(j, self.ny) + ramp(k, self.nz);
            (-gamma * dt).exp()
        });
        self.damping_decay = Some(decay);
    }

    /// 1-D convenience constructor (`ny = nz = 1`).
    /// # Errors
    /// - Propagates [`Self::new`] validation failures.
    pub fn new_1d(
        n: usize,
        dx: f64,
        dt: f64,
        rho: f64,
        m_inf: f64,
        arms: &[(f64, f64)],
    ) -> KwaversResult<Self> {
        Self::new(n, 1, 1, dx, 1.0, 1.0, dt, rho, m_inf, arms)
    }

    /// Build from a [`GeneralizedMaxwellModel`] (its `M_∞`, arms, and density)
    /// plus the grid and time step.
    /// # Errors
    /// - Propagates [`Self::new`] validation failures.
    #[allow(clippy::too_many_arguments)]
    pub fn from_generalized_maxwell(
        model: &GeneralizedMaxwellModel,
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        dt: f64,
    ) -> KwaversResult<Self> {
        Self::new(
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            dt,
            model.density(),
            model.equilibrium_modulus(),
            model.arms(),
        )
    }

    /// Build from a **heterogeneous power-law medium**: per-voxel density
    /// `ρ(x)`, phase velocity `c(x)` at `f_ref`, absorption coefficient `α(x)`
    /// \[Np·m⁻¹] at `f_ref`, and **per-voxel power-law exponent `γ(x)`**, so
    /// `α(x, f) = α(x)·(f/f_ref)^{γ(x)}`.
    ///
    /// A shared log-spaced relaxation-time grid is fitted per voxel by
    /// [`fit_power_law_fields`]: one non-negative strength vector per voxel on
    /// one `τₗ` grid, so a spatially varying exponent costs no extra memory
    /// fields. The fit also calibrates each voxel's equilibrium modulus so the
    /// **dispersive** phase velocity at `f_ref` equals `c(x)` — the naive
    /// `M_∞ = ρc²` runs fast by the Kramers–Krönig dispersion increment.
    ///
    /// This is the bridge from the `HuAcousticModel`/`CtMediumBuilder` tissue
    /// pipeline (book §4.5) to the broadband solver.
    ///
    /// # Errors
    /// - Field shape ≠ grid, non-positive `ρ`/`c`, negative `α`, a band the fit
    ///   rejects, or `n_arms == 0`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_power_law_fields(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        dt: f64,
        rho: &Array3<f64>,
        c: &Array3<f64>,
        alpha_np_m: &Array3<f64>,
        exponent: &Array3<f64>,
        f_min: f64,
        f_max: f64,
        n_arms: usize,
        f_ref: f64,
    ) -> KwaversResult<Self> {
        let shape = [nx, ny, nz];
        let positive = |a: &Array3<f64>| a.shape() == shape && a.iter().all(|&v| v > 0.0);
        if !positive(rho) || !positive(c) || alpha_np_m.shape() != shape {
            return Err(KwaversError::InvalidInput(
                "ρ, c, α fields must be grid-shaped (ρ, c positive)".to_owned(),
            ));
        }
        if exponent.shape() != shape {
            return Err(KwaversError::InvalidInput(
                "γ field must be grid-shaped".to_owned(),
            ));
        }

        let mut band = FitBand::new(f_min, f_max, n_arms)?;
        band.n_samples = (8 * n_arms).max(64);
        let fit = fit_power_law_fields(alpha_np_m, exponent, c, rho, f_ref, &band)?;
        let arms = fit.arm_fields();

        Self::new_heterogeneous(
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            dt,
            rho,
            fit.equilibrium_modulus(),
            &arms,
        )
    }

    /// Maximum unrelaxed (high-frequency) sound speed `max √(M_U(x)/ρ(x))`
    /// \[m·s⁻¹] over the grid — the CFL reference speed.
    #[must_use]
    pub fn unrelaxed_speed(&self) -> f64 {
        self.max_unrelaxed_speed
    }

    /// Pressure field \\[Pa\].
    #[must_use]
    pub fn pressure(&self) -> &Array3<f64> {
        &self.p
    }

    /// Overwrite the pressure field (shape must match the grid); resets velocity
    /// and memory variables to zero. Used to set an initial condition.
    /// # Errors
    /// - Shape mismatch.
    pub fn set_pressure(&mut self, p: &Array3<f64>) -> KwaversResult<()> {
        if p.shape() != [self.nx, self.ny, self.nz] {
            return Err(KwaversError::InvalidInput(
                "pressure shape must equal the grid".to_owned(),
            ));
        }
        self.p.assign(p);
        if self.axes.x {
            self.vx.fill(0.0);
        }
        if self.axes.y {
            self.vy.fill(0.0);
        }
        if self.axes.z {
            self.vz.fill(0.0);
        }
        for s in &mut self.sigma {
            s.fill(0.0);
        }
        // Restart the simulation clock and sensor traces (sources/sensors kept).
        self.step_count = 0;
        for trace in &mut self.sensor_record {
            trace.clear();
        }
        Ok(())
    }

    /// Acoustic energy `Σ [p²/(2M_∞) + ρ|v|²/2] ΔV` \`J`. Conserved (to leapfrog
    /// round-off) for the lossless medium; decays monotonically with relaxation.
    #[must_use]
    ///
    /// # Panics
    ///
    /// Panics if a caller-supplied shape or an internal solver state violates
    /// the precondition required by this operation.
    pub fn energy(&self) -> f64 {
        // PE = Σ p²/(2 M_∞(x));  KE = Σ ρ(x)|v|²/2 = Σ |v|²/(2/ρ) = Σ |v|²·inv_rho⁻¹/2.
        let pe = leto_ops::zip_fold(
            &self.p.view(),
            &self.m_inf.view(),
            0.0_f64,
            |acc, &p, &mi| acc + p * p / (2.0 * mi),
        )
        .expect("invariant: pressure and modulus fields share grid shape");
        // Inactive-axis velocity components are exactly zero for all time, so
        // their kinetic contribution is the zero term and is not traversed.
        let vx_iter = self.vx.iter().copied().chain(std::iter::repeat(0.0_f64));
        let vy_iter = self.vy.iter().copied().chain(std::iter::repeat(0.0_f64));
        let vz_iter = self.vz.iter().copied().chain(std::iter::repeat(0.0_f64));
        let ke = vx_iter
            .zip(vy_iter)
            .zip(vz_iter)
            .zip(self.inv_rho.iter())
            .fold(0.0_f64, |acc, (((vx, vy), vz), &ir)| {
                acc + (vx * vx + vy * vy + vz * vz) / (2.0 * ir)
            });
        (pe + ke) * self.cell_volume
    }

    /// Advance the state by one time step `Δt`.
    ///
    /// # Panics
    ///
    /// Panics if a caller-supplied shape or an internal solver state violates
    /// the precondition required by this operation.
    pub fn step(&mut self) {
        // 1. Velocity half-step: v += -(Δt/ρ(x)) ∇p (per component, per voxel).
        // Staging slots are assigned in canonical axis order — first active
        // axis → `gx`, second → `gy`, third → `gz` — so every layout stages
        // into grid-shaped buffers only (`gz` exists exactly in the layout
        // whose step needs a third slot) and each slot is consumed by its
        // velocity update before the divergence pass restages it. An inactive
        // axis stages no derivative and updates no velocity: its component is
        // the exact positive zero for all time (the value the full-storage
        // layout wrote and then added zero to).
        let dt = self.dt;
        let axes = self.axes;
        let slot_x: Option<usize> = axes.x.then_some(0);
        let slot_y: Option<usize> = axes.y.then_some(usize::from(axes.x));
        let slot_z: Option<usize> = axes.z.then_some(usize::from(axes.x) + usize::from(axes.y));
        {
            let buffers: [&mut Array3<f64>; 3] = [&mut self.gx, &mut self.gy, &mut self.gz];
            if let Some(s) = slot_x {
                Self::axis_derivative(
                    &self.fft,
                    &self.kx,
                    0,
                    &self.p,
                    &mut self.cbuf,
                    &mut *buffers[s],
                );
                leto_ops::zip_mut_with(
                    self.vx.view_mut(),
                    (&buffers[s].view(), &self.inv_rho.view()),
                    |v, (&g, &ir)| *v += -dt * ir * g,
                )
                .expect("invariant: velocity-x update fields share grid shape");
            }
            if let Some(s) = slot_y {
                Self::axis_derivative(
                    &self.fft,
                    &self.ky,
                    1,
                    &self.p,
                    &mut self.cbuf,
                    &mut *buffers[s],
                );
                leto_ops::zip_mut_with(
                    self.vy.view_mut(),
                    (&buffers[s].view(), &self.inv_rho.view()),
                    |v, (&g, &ir)| *v += -dt * ir * g,
                )
                .expect("invariant: velocity-y update fields share grid shape");
            }
            if let Some(s) = slot_z {
                Self::axis_derivative(
                    &self.fft,
                    &self.kz,
                    2,
                    &self.p,
                    &mut self.cbuf,
                    &mut *buffers[s],
                );
                leto_ops::zip_mut_with(
                    self.vz.view_mut(),
                    (&buffers[s].view(), &self.inv_rho.view()),
                    |v, (&g, &ir)| *v += -dt * ir * g,
                )
                .expect("invariant: velocity-z update fields share grid shape");
            }

            // 2. Dilatation rate D = ∇·v, staged into the same slots: the
            // first active axis's derivative lands in `gx` — the divergence
            // output — and the rest in `gy`/`gz`. A missing axis contributes
            // its exact positive-zero derivative (the identity the
            // full-storage layout derives from the zero velocity and the
            // singleton wavenumber).
            if let Some(s) = slot_x {
                Self::axis_derivative(
                    &self.fft,
                    &self.kx,
                    0,
                    &self.vx,
                    &mut self.cbuf,
                    &mut *buffers[s],
                );
            }
            if let Some(s) = slot_y {
                Self::axis_derivative(
                    &self.fft,
                    &self.ky,
                    1,
                    &self.vy,
                    &mut self.cbuf,
                    &mut *buffers[s],
                );
            }
            if let Some(s) = slot_z {
                Self::axis_derivative(
                    &self.fft,
                    &self.kz,
                    2,
                    &self.vz,
                    &mut self.cbuf,
                    &mut *buffers[s],
                );
            }
        }

        // gx = ∂vx/∂x + (∂vy/∂y + ∂vz/∂z): the reference association, in one
        // flat combine pass per layout. With a single active axis the staged
        // derivative already is the divergence; with none it is the zero fill.
        match axes.count() {
            0 => self.gx.fill(0.0),
            1 => {}
            2 => leto_ops::zip_mut_with(self.gx.view_mut(), &self.gy.view(), |d_, &y| {
                *d_ += y;
            })
            .expect("invariant: divergence components share grid shape"),
            _ => leto_ops::zip_mut_with(
                self.gx.view_mut(),
                (&self.gy.view(), &self.gz.view()),
                |d_, (&y, &z)| *d_ += y + z,
            )
            .expect("invariant: divergence components share grid shape"),
        }

        // 3. Advance each σ_l with the exact exponential integrator (per-voxel
        //    coefficients) and accumulate its trapezoidal pressure contribution
        //    into gy (reused as the relax sum): σ_new = decay·σ + gain·D.
        self.gy.fill(0.0);
        // Two mutable outputs (`sigma` arm and `self.gy` relax-sum) updated in
        // lockstep with 4 read inputs; leto_ops zip is single-lhs only, so this
        // is a native flat-index loop over the contiguous same-shape grid fields.
        for (arm, sigma) in self.arms.iter().zip(self.sigma.iter_mut()) {
            let s_slice = sigma
                .as_slice_mut()
                .expect("invariant: viscoacoustic σ arm is contiguous");
            let gx_slice = self
                .gx
                .as_slice()
                .expect("invariant: viscoacoustic gx is contiguous");
            let gy_slice = self
                .gy
                .as_slice_mut()
                .expect("invariant: viscoacoustic gy is contiguous");
            let decay_slice = arm
                .decay
                .as_slice()
                .expect("invariant: viscoacoustic decay is contiguous");
            let gain_slice = arm
                .gain
                .as_slice()
                .expect("invariant: viscoacoustic gain is contiguous");
            let inv_tau_slice = arm
                .inv_tau
                .as_slice()
                .expect("invariant: viscoacoustic inv_tau is contiguous");
            for idx in 0..s_slice.len() {
                let old = s_slice[idx];
                let new = decay_slice[idx].mul_add(old, gain_slice[idx] * gx_slice[idx]);
                gy_slice[idx] += 0.5 * (old + new) * inv_tau_slice[idx];
                s_slice[idx] = new;
            }
        }

        // 4. Pressure update: p += -Δt (M_U(x) D + Σ_l ½(σ_l+σ_l^new)/τ_l(x)).
        leto_ops::zip_mut_with(
            self.p.view_mut(),
            (&self.gx.view(), &self.gy.view(), &self.m_u.view()),
            |p, (&d, &relax, &mu)| *p -= dt * (mu * d + relax),
        )
        .expect("invariant: pressure update fields share grid shape");

        // 5. Soft pressure sources: p[index] += signal[step].
        for (index, signal) in &self.pressure_sources {
            if let Some(&s) = signal.get(self.step_count) {
                self.p[[index.0, index.1, index.2]] += s;
            }
        }

        // 6. Absorbing boundary: damp p and v inside the sponge layer. Each
        // active axis is damped exactly once here (the removed inactive-axis
        // pass added only a multiply by one over a zero field).
        if let Some(decay) = &self.damping_decay {
            leto_ops::zip_mut_with(self.p.view_mut(), &decay.view(), |p, &d| *p *= d)
                .expect("invariant: pressure and damping fields share grid shape");
            if axes.x {
                leto_ops::zip_mut_with(self.vx.view_mut(), &decay.view(), |v, &d| *v *= d)
                    .expect("invariant: velocity-x and damping fields share grid shape");
            }
            if axes.y {
                leto_ops::zip_mut_with(self.vy.view_mut(), &decay.view(), |v, &d| *v *= d)
                    .expect("invariant: velocity-y and damping fields share grid shape");
            }
            if axes.z {
                leto_ops::zip_mut_with(self.vz.view_mut(), &decay.view(), |v, &d| *v *= d)
                    .expect("invariant: velocity-z and damping fields share grid shape");
            }
        }

        // 7. Record sensor traces, then advance the simulation clock.
        for (trace, &index) in self
            .sensor_record
            .iter_mut()
            .zip(self.pressure_sensors.iter())
        {
            trace.push(self.p[[index.0, index.1, index.2]]);
        }
        self.step_count += 1;
    }
}

/// FFT-order signed wavenumbers `k\[m\] = 2π m'/(n·Δx)` with `m' = m` for `m < n/2`
/// and `m' = m − n` otherwise. For `n = 1` this is `[0]` (derivative along a
/// singleton axis is zero).
///
/// The Nyquist bin (`m = n/2`, even `n`) is forced to 0: the first-derivative
/// operator `i·k` is purely imaginary there, so a nonzero Nyquist wavenumber
/// would inject a spurious antisymmetric (non-real) component into `∂p/∂α` of a
/// real field. Zeroing it guarantees a real-valued spectral derivative, matching
/// the KZK τ-derivative convention (`kzk::nonlinearity`).
fn fft_wavenumbers(n: usize, dx: f64) -> Vec<f64> {
    let norm = TWO_PI / (n as f64 * dx);
    (0..n)
        .map(|m| {
            if n.is_multiple_of(2) && m == n / 2 {
                return 0.0;
            }
            let signed = if m < n / 2 {
                m as f64
            } else {
                m as f64 - n as f64
            };
            signed * norm
        })
        .collect()
}
