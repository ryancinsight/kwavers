//! Velocity update kernels (scalar, AVX2) for [`FdtdSimdOps`].

use super::FdtdSimdOps;

impl FdtdSimdOps {
    /// Scalar velocity update.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn update_velocity_scalar(
        &self,
        velocity: &mut [f32],
        velocity_prev: &[f32],
        pressure_gradient: &[f32],
        dt_over_rho: f32,
        nx: usize,
        ny: usize,
        nz: usize,
    ) {
        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let idx = i + j * nx + k * nx * ny;
                    velocity[idx] =
                        dt_over_rho.mul_add(-pressure_gradient[idx], velocity_prev[idx]);
                }
            }
        }
    }

    /// AVX2 velocity update.
    ///
    /// Evaluates v^{n+1} = v^n − (Δt/ρ) · ∇p on 8-wide f32 lanes.
    ///
    /// # Safety
    ///
    /// - CPU detection ensures AVX2 support before this method is called.
    /// - Loop guard `i + 7 < nx - 1` keeps 8-lane loads in the row interior.
    /// - `_mm256_loadu_ps` / `_mm256_storeu_ps` handle unaligned addresses.
    ///
    /// **Preconditions**: all slices have length ≥ nx · ny · nz; interior loops only.
    #[allow(clippy::too_many_arguments)]
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[allow(unsafe_code)]
    pub(super) unsafe fn update_velocity_avx2(
        &self,
        velocity: &mut [f32],
        velocity_prev: &[f32],
        pressure_gradient: &[f32],
        dt_over_rho: f32,
        nx: usize,
        ny: usize,
        nz: usize,
    ) {
        use std::arch::x86_64::{
            _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps, _mm256_sub_ps,
        };

        let dt_rho = _mm256_set1_ps(dt_over_rho);

        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                let mut i = 1;
                while i + 7 < nx - 1 {
                    let idx = i + j * nx + k * nx * ny;

                    let v_prev = _mm256_loadu_ps(velocity_prev.as_ptr().add(idx));
                    let grad_p = _mm256_loadu_ps(pressure_gradient.as_ptr().add(idx));

                    // v_prev - (Δt/ρ) * ∇p
                    let result = _mm256_sub_ps(v_prev, _mm256_mul_ps(dt_rho, grad_p));

                    _mm256_storeu_ps(velocity.as_mut_ptr().add(idx), result);
                    i += 8;
                }

                // Scalar tail
                while i < nx - 1 {
                    let idx = i + j * nx + k * nx * ny;
                    velocity[idx] =
                        dt_over_rho.mul_add(-pressure_gradient[idx], velocity_prev[idx]);
                    i += 1;
                }
            }
        }
    }
}
