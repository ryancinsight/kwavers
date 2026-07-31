//! Velocity update kernel backed by `hermes-simd` runtime dispatch.
//!
//! Implements the staggered-grid velocity update
//!
//! ```text
//! v^{n+1} = v^n − (Δt/ρ)·∇p^n
//! ```
//!
//! which is an AXPY operation: `y += α·x` with α = −Δt/ρ and y initialised
//! from `v^n`.  All unsafe intrinsics are in `hermes_simd_intrinsics`.

use super::FdtdSimdOps;
use hermes_simd::axpy;

impl FdtdSimdOps {
    /// Hermes-dispatched 3-D FDTD velocity update.
    ///
    /// For each interior row the update is:
    ///   1. `velocity[row] ← velocity_prev[row]`
    ///   2. `velocity[row] += −(Δt/ρ) · pressure_gradient[row]`
    #[allow(clippy::too_many_arguments)]
    pub(super) fn update_velocity_hermes(
        &self,
        velocity: &mut [f32],
        velocity_prev: &[f32],
        pressure_gradient: &[f32],
        dt_over_rho: f32,
        nx: usize,
        ny: usize,
        nz: usize,
    ) {
        let row_len = nx - 2;

        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                let row_start = 1 + j * nx + k * nx * ny;
                let row_end = row_start + row_len;

                // v[row] = v_prev[row]
                velocity[row_start..row_end].copy_from_slice(&velocity_prev[row_start..row_end]);

                // v[row] += −(Δt/ρ) · ∇p[row]
                axpy(
                    -dt_over_rho,
                    &pressure_gradient[row_start..row_end],
                    &mut velocity[row_start..row_end],
                )
                .expect("velocity axpy: length mismatch");
            }
        }
    }
}
