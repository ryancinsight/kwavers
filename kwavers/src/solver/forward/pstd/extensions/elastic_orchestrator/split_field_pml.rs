//! Bérenger (1994) split-field PML for the elastic PSTD orchestrator.
//!
//! # Theorem (split-field PML exactness)
//!
//! For an elastic medium with velocity–stress formulation, each velocity
//! component `v_α` is split into three directional sub-fields
//! `v_{α,x}`, `v_{α,y}`, `v_{α,z}` satisfying independent ODEs:
//!
//! ```text
//!   ∂_t v_{α,β} + σ_β(x_β) · v_{α,β} = (1/ρ) ∂_β σ_{αβ}     (α, β ∈ {x,y,z})
//!   v_α = v_{α,x} + v_{α,y} + v_{α,z}
//! ```
//!
//! and each stress component `σ_{αβ}` is split into contributions from each
//! velocity-gradient direction. The damping profile `σ_β(x_β) ≥ 0` is zero
//! in the interior and increases monotonically through the absorbing layer of
//! thickness `L_β`. This formulation is perfectly matched to outgoing waves at
//! **all angles of incidence and all frequencies**, yielding zero theoretical
//! reflection (Bérenger 1994, Collino & Tsogka 2001). The simple
//! real-space-exponential PML in [`super::pml`] achieves the same σ_max
//! profile but does not split the fields, producing non-zero reflection at
//! oblique incidence.
//!
//! # Discrete integrator
//!
//! The exact solution to `∂_t f + σf = g` over one step dt is
//!
//! ```text
//!   f(t+dt) = exp(−σ·dt) · f(t) + β · g(t+dt/2)
//! ```
//!
//! where `β = (1 − exp(−σ·dt)) / σ` for σ > 0 (reduces to dt at σ = 0
//! by L'Hôpital), i.e., the standard leapfrog update in the interior.
//! `α = exp(−σ·dt)` and `β` are precomputed per cell once at construction;
//! β absorbs the `dt` factor so the RHS need not carry it.
//!
//! # Memory
//!
//! `SplitFieldState` stores 24 real-valued `Array3<f64>` sub-fields per grid:
//! - 9 velocity sub-fields: `v_{α,x}`, `v_{α,y}`, `v_{α,z}` for α ∈ {x,y,z}
//! - 15 stress sub-fields: normal-stress directions (3×3) + shear-stress
//!   axis pairs (3×2 = 6 non-zero off-diagonal contributions)
//!
//! At 64×64×64 this is ≈ 503 MB. The `Box<SplitFieldState>` in the
//! orchestrator avoids stack overflow.
//!
//! # References
//!
//! - Bérenger J. P. (1994). A perfectly matched layer for the absorption of
//!   electromagnetic waves. J. Comput. Phys. **114**(2), 185–200.
//! - Collino F. & Tsogka C. (2001). Application of the perfectly matched
//!   absorbing layer model to the linear elastodynamic problem in anisotropic
//!   heterogeneous media. Geophysics **66**(1), 294–307.

use super::pml::build_axis_alpha_beta;
use ndarray::{Array1, Array3};

/// Pre-computed per-axis exponential and integration coefficients for the
/// split-field PML.
///
/// - `alpha_α[i] = exp(−σ_α(i)·dt)` ∈ (0, 1]: multiplicative decay factor
/// - `beta_α[i]  = (1 − alpha_α[i]) / σ_α(i)` for σ > 0, else dt: RHS
///   integration factor (absorbs the time-step in the exact integrator)
#[derive(Debug, Clone)]
pub struct ElasticSplitFieldPml {
    alpha_x: Array1<f64>,
    beta_x: Array1<f64>,
    alpha_y: Array1<f64>,
    beta_y: Array1<f64>,
    alpha_z: Array1<f64>,
    beta_z: Array1<f64>,
    /// Absorbing-layer thickness in cells per axis.
    thickness_cells: (usize, usize, usize),
}

impl ElasticSplitFieldPml {
    /// Build the split-field PML for a `(nx, ny, nz)` grid.
    ///
    /// Parameters follow the same convention as [`super::pml::ElasticPml::new`]:
    /// `c_max` is the maximum wave speed in the medium, `r0` is the target
    /// theoretical reflection coefficient, and the polynomial order p = 4
    /// (Roden & Gedney 2000 optimal for spectral solvers) is fixed.
    #[must_use]
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        thickness_cells: (usize, usize, usize),
        dx: f64,
        dy: f64,
        dz: f64,
        c_max: f64,
        dt: f64,
        r0: f64,
    ) -> Self {
        const P: f64 = 4.0;
        let (alpha_x, beta_x) =
            build_axis_alpha_beta(nx, thickness_cells.0, dx, c_max, dt, r0, P);
        let (alpha_y, beta_y) =
            build_axis_alpha_beta(ny, thickness_cells.1, dy, c_max, dt, r0, P);
        let (alpha_z, beta_z) =
            build_axis_alpha_beta(nz, thickness_cells.2, dz, c_max, dt, r0, P);
        Self {
            alpha_x,
            beta_x,
            alpha_y,
            beta_y,
            alpha_z,
            beta_z,
            thickness_cells,
        }
    }

    /// Borrow the x-axis `(alpha, beta)` coefficient arrays.
    #[must_use]
    pub fn x_coeffs(&self) -> (&Array1<f64>, &Array1<f64>) {
        (&self.alpha_x, &self.beta_x)
    }

    /// Borrow the y-axis `(alpha, beta)` coefficient arrays.
    #[must_use]
    pub fn y_coeffs(&self) -> (&Array1<f64>, &Array1<f64>) {
        (&self.alpha_y, &self.beta_y)
    }

    /// Borrow the z-axis `(alpha, beta)` coefficient arrays.
    #[must_use]
    pub fn z_coeffs(&self) -> (&Array1<f64>, &Array1<f64>) {
        (&self.alpha_z, &self.beta_z)
    }

    /// Absorbing-layer thickness in cells per axis.
    #[must_use]
    pub fn thickness_cells(&self) -> (usize, usize, usize) {
        self.thickness_cells
    }
}

/// Persistent sub-field state for the Bérenger split-field PML.
///
/// All 24 fields start at zero (quiescent initial conditions) and evolve
/// each step via the exact discrete integrator. In the interior (σ = 0),
/// the sub-fields sum to the standard leapfrog result; in the PML, each
/// sub-field decays independently along its axis.
#[derive(Debug)]
pub struct SplitFieldState {
    // ── Velocity sub-fields ────────────────────────────────────────────────
    // `v_{α,β}`: x-velocity (α=x) driven by stress-divergence along axis β.
    pub vxx: Array3<f64>,
    pub vxy: Array3<f64>,
    pub vxz: Array3<f64>,
    // y-velocity
    pub vyx: Array3<f64>,
    pub vyy: Array3<f64>,
    pub vyz: Array3<f64>,
    // z-velocity
    pub vzx: Array3<f64>,
    pub vzy: Array3<f64>,
    pub vzz: Array3<f64>,

    // ── Stress sub-fields ──────────────────────────────────────────────────
    // Normal stress σ_xx split by driving direction:
    //   txx_x ← (λ+2μ) ∂_x vx,  txx_y ← λ ∂_y vy,  txx_z ← λ ∂_z vz
    pub txx_x: Array3<f64>,
    pub txx_y: Array3<f64>,
    pub txx_z: Array3<f64>,
    // Normal stress σ_yy:
    //   tyy_x ← λ ∂_x vx,  tyy_y ← (λ+2μ) ∂_y vy,  tyy_z ← λ ∂_z vz
    pub tyy_x: Array3<f64>,
    pub tyy_y: Array3<f64>,
    pub tyy_z: Array3<f64>,
    // Normal stress σ_zz:
    //   tzz_x ← λ ∂_x vx,  tzz_y ← λ ∂_y vy,  tzz_z ← (λ+2μ) ∂_z vz
    pub tzz_x: Array3<f64>,
    pub tzz_y: Array3<f64>,
    pub tzz_z: Array3<f64>,
    // Shear stress σ_xy (no z-contribution in isotropic media):
    //   txy_x ← μ ∂_x vy,  txy_y ← μ ∂_y vx
    pub txy_x: Array3<f64>,
    pub txy_y: Array3<f64>,
    // Shear stress σ_xz (no y-contribution):
    //   txz_x ← μ ∂_x vz,  txz_z ← μ ∂_z vx
    pub txz_x: Array3<f64>,
    pub txz_z: Array3<f64>,
    // Shear stress σ_yz (no x-contribution):
    //   tyz_y ← μ ∂_y vz,  tyz_z ← μ ∂_z vy
    pub tyz_y: Array3<f64>,
    pub tyz_z: Array3<f64>,
}

impl SplitFieldState {
    /// Allocate all 24 sub-fields zeroed to `(nx, ny, nz)`.
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        macro_rules! zero {
            () => {
                Array3::<f64>::zeros((nx, ny, nz))
            };
        }
        Self {
            vxx: zero!(),
            vxy: zero!(),
            vxz: zero!(),
            vyx: zero!(),
            vyy: zero!(),
            vyz: zero!(),
            vzx: zero!(),
            vzy: zero!(),
            vzz: zero!(),
            txx_x: zero!(),
            txx_y: zero!(),
            txx_z: zero!(),
            tyy_x: zero!(),
            tyy_y: zero!(),
            tyy_z: zero!(),
            tzz_x: zero!(),
            tzz_y: zero!(),
            tzz_z: zero!(),
            txy_x: zero!(),
            txy_y: zero!(),
            txz_x: zero!(),
            txz_z: zero!(),
            tyz_y: zero!(),
            tyz_z: zero!(),
        }
    }
}
