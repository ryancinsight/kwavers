//! Pre-allocated workspace for one elastic velocity-Verlet time step.
//!
//! ## Motivation
//!
//! Each call to `TimeIntegrator::step` triggers two `stress_divergence_into`
//! invocations (acceleration at tⁿ and at tⁿ⁺¹) plus three acceleration
//! arrays.  Without pre-allocation, each `stress_divergence_into` allocates
//! 9 `Array3<f64>` (6 stress + 3 divergence) and `step` allocates 3 more
//! (ax, ay, az) — 21 allocations per step, 24 total because `step` calls
//! `compute_acceleration` twice.  For a 128³ grid at f64:
//!
//! ```text
//! 24 × 128³ × 8 B = 24 × 16 MiB = 384 MiB of heap activity per step
//! ```
//!
//! `ElasticStepScratch` pre-allocates all 12 grid workspaces and three PML
//! axis-factor arrays **once** before the time loop, reducing per-step heap
//! activity to zero.
//!
//! ## Theorem (no aliasing)
//!
//! The 12 grid fields and three axis fields are independent allocations; no
//! two fields alias the same memory region. `stress_divergence_into` writes
//! `{sxx,syy,szz,sxy,sxz,syz,div_x,div_y,div_z}` and reads nothing from
//! scratch → race-free parallel writes.  `compute_acceleration` subsequently
//! reads `{div_x,div_y,div_z}` (immutable views) and writes `{ax,ay,az}`
//! (mutable views) — disjoint field sets, safe under Rust NLL field-split
//! borrows.  The velocity-Verlet update reads `{ax,ay,az}` immutably and
//! writes `{vx,vy,vz}` on the wave field — separate allocation entirely.

use leto::{Array1, Array3};

/// Reusable scratch arrays for one `TimeIntegrator` velocity-Verlet step.
///
/// Construct once before the time loop with [`ElasticStepScratch::new`];
/// pass `&mut scratch` to every
/// [`crate::forward::elastic::swe::integration::integrator::TimeIntegrator::step`] or
/// [`crate::forward::elastic::swe::integration::integrator::TimeIntegrator::step_with_body_forces`]
/// call.
///
/// **Do not** construct inside the time loop — that defeats the purpose and
/// restores the per-step allocation cost.
#[derive(Debug)]
pub struct ElasticStepScratch {
    // --- Pass 1a: diagonal stress components ---
    /// σxx = (λ+2μ)εxx + λ(εyy+εzz)
    pub sxx: Array3<f64>,
    /// σyy = (λ+2μ)εyy + λ(εxx+εzz)
    pub syy: Array3<f64>,
    /// σzz = (λ+2μ)εzz + λ(εxx+εyy)
    pub szz: Array3<f64>,
    // --- Pass 1b: off-diagonal stress components ---
    /// σxy = σyx = μ(∂ux/∂y + ∂uy/∂x)
    pub sxy: Array3<f64>,
    /// σxz = σzx = μ(∂ux/∂z + ∂uz/∂x)
    pub sxz: Array3<f64>,
    /// σyz = σzy = μ(∂uy/∂z + ∂uz/∂y)
    pub syz: Array3<f64>,
    // --- Pass 2: stress tensor divergence ---
    /// (∇·σ)_x = ∂σxx/∂x + ∂σxy/∂y + ∂σxz/∂z
    pub div_x: Array3<f64>,
    /// (∇·σ)_y = ∂σxy/∂x + ∂σyy/∂y + ∂σyz/∂z
    pub div_y: Array3<f64>,
    /// (∇·σ)_z = ∂σxz/∂x + ∂σyz/∂y + ∂σzz/∂z
    pub div_z: Array3<f64>,
    // --- Acceleration: a = (∇·σ + f) / ρ ---
    /// x-component of elastic acceleration
    pub ax: Array3<f64>,
    /// y-component of elastic acceleration
    pub ay: Array3<f64>,
    /// z-component of elastic acceleration
    pub az: Array3<f64>,
    /// Cached x-axis PML factors for the active time step.
    pml_x: Array1<f64>,
    /// Cached y-axis PML factors for the active time step.
    pml_y: Array1<f64>,
    /// Cached z-axis PML factors for the active time step.
    pml_z: Array1<f64>,
    /// Time step used to derive the cached PML factors.
    pml_dt: Option<f64>,
}

impl ElasticStepScratch {
    /// Allocate 12 grid workspaces and three PML axis-factor arrays.
    ///
    /// Cost: `8 × (12 × nx × ny × nz + nx + ny + nz)` bytes, paid once
    /// before the time loop. For 128³: about 192 MiB one-time; zero per-step
    /// allocation thereafter.
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            sxx: Array3::<f64>::zeros((nx, ny, nz)),
            syy: Array3::<f64>::zeros((nx, ny, nz)),
            szz: Array3::<f64>::zeros((nx, ny, nz)),
            sxy: Array3::<f64>::zeros((nx, ny, nz)),
            sxz: Array3::<f64>::zeros((nx, ny, nz)),
            syz: Array3::<f64>::zeros((nx, ny, nz)),
            div_x: Array3::<f64>::zeros((nx, ny, nz)),
            div_y: Array3::<f64>::zeros((nx, ny, nz)),
            div_z: Array3::<f64>::zeros((nx, ny, nz)),
            ax: Array3::<f64>::zeros((nx, ny, nz)),
            ay: Array3::<f64>::zeros((nx, ny, nz)),
            az: Array3::<f64>::zeros((nx, ny, nz)),
            pml_x: Array1::<f64>::zeros(nx),
            pml_y: Array1::<f64>::zeros(ny),
            pml_z: Array1::<f64>::zeros(nz),
            pml_dt: None,
        }
    }

    /// Return separable PML factors for `dt`, recomputing only when it changes.
    pub(crate) fn pml_factors(
        &mut self,
        sigma_x: &Array1<f64>,
        sigma_y: &Array1<f64>,
        sigma_z: &Array1<f64>,
        dt: f64,
    ) -> (&[f64], &[f64], &[f64]) {
        if self.pml_dt != Some(dt) {
            fill_pml_axis(&mut self.pml_x, sigma_x, dt);
            fill_pml_axis(&mut self.pml_y, sigma_y, dt);
            fill_pml_axis(&mut self.pml_z, sigma_z, dt);
            self.pml_dt = Some(dt);
        }

        (
            self.pml_x
                .as_slice()
                .expect("invariant: x-axis PML factors use standard layout"),
            self.pml_y
                .as_slice()
                .expect("invariant: y-axis PML factors use standard layout"),
            self.pml_z
                .as_slice()
                .expect("invariant: z-axis PML factors use standard layout"),
        )
    }
}

fn fill_pml_axis(factors: &mut Array1<f64>, sigma: &Array1<f64>, dt: f64) {
    let factors = factors
        .as_slice_mut()
        .expect("invariant: PML factors use standard layout");
    let sigma = sigma
        .as_slice()
        .expect("invariant: PML sigma profiles use standard layout");
    debug_assert_eq!(factors.len(), sigma.len());

    for (factor, &coefficient) in factors.iter_mut().zip(sigma) {
        *factor = (-coefficient * dt).exp();
    }
}
