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
//! `ElasticStepScratch` pre-allocates all 12 workspace arrays **once** before
//! the time loop, reducing per-step heap activity to zero.
//!
//! ## Theorem (no aliasing)
//!
//! The 12 fields are independent `Array3` allocations; no two fields alias the
//! same memory region.  `stress_divergence_into` writes
//! `{sxx,syy,szz,sxy,sxz,syz,div_x,div_y,div_z}` and reads nothing from
//! scratch → race-free parallel writes.  `compute_acceleration` subsequently
//! reads `{div_x,div_y,div_z}` (immutable views) and writes `{ax,ay,az}`
//! (mutable views) — disjoint field sets, safe under Rust NLL field-split
//! borrows.  The velocity-Verlet update reads `{ax,ay,az}` immutably and
//! writes `{vx,vy,vz}` on the wave field — separate allocation entirely.

use ndarray::Array3;

/// Reusable scratch arrays for one `TimeIntegrator` velocity-Verlet step.
///
/// Construct once before the time loop with [`ElasticStepScratch::new`];
/// pass `&mut scratch` to every [`TimeIntegrator::step`] or
/// [`TimeIntegrator::step_with_body_forces`] call.
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
}

impl ElasticStepScratch {
    /// Allocate all 12 workspace arrays with shape `(nx, ny, nz)`.
    ///
    /// Cost: `12 × nx × ny × nz × 8` bytes, paid once before the time loop.
    /// For 128³: 192 MiB one-time; zero per-step allocation thereafter.
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
        }
    }
}
