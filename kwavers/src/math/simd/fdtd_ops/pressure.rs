//! Pressure update kernels (scalar, AVX2, AVX-512) for [`FdtdSimdOps`].

use super::FdtdSimdOps;

impl FdtdSimdOps {
    /// Scalar fallback for pressure update.
    pub(super) fn update_pressure_scalar(
        &self,
        pressure: &mut [f32],
        pressure_prev: &[f32],
        laplacian: &[f32],
        c_squared_dt_squared: f32,
        nx: usize,
        ny: usize,
        nz: usize,
    ) {
        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let idx = i + j * nx + k * nx * ny;
                    pressure[idx] = c_squared_dt_squared.mul_add(
                        laplacian[idx],
                        2.0f32.mul_add(pressure[idx], -pressure_prev[idx]),
                    );
                }
            }
        }
    }

    /// AVX2-optimized pressure update.
    ///
    /// Evaluates p^{n+1} = 2p^n - p^{n-1} + c²Δt² · Lap(p^n) on 8-wide f32 lanes.
    ///
    /// # Safety
    ///
    /// - CPU feature detection via `SimdConfig::detect()` ensures AVX2 support.
    /// - Pointer arithmetic idx = i + j·nx + k·nx·ny is bounded by loop invariants.
    /// - `_mm256_loadu_ps` / `_mm256_storeu_ps` handle unaligned Rust slice addresses.
    /// - 8-lane boundary check (`i + 7 < nx - 1`) prevents out-of-bounds reads/writes.
    ///
    /// **Preconditions**:
    /// - All slices have length ≥ nx · ny · nz.
    /// - Loops cover interior points only (1 ≤ i, j, k < nx-1, ny-1, nz-1).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[allow(unsafe_code)]
    pub(super) unsafe fn update_pressure_avx2(
        &self,
        pressure: &mut [f32],
        pressure_prev: &[f32],
        laplacian: &[f32],
        c_squared_dt_squared: f32,
        nx: usize,
        ny: usize,
        nz: usize,
    ) {
        use std::arch::x86_64::{
            _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps,
            _mm256_sub_ps,
        };

        let two = _mm256_set1_ps(2.0);
        let c_dt2 = _mm256_set1_ps(c_squared_dt_squared);

        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                let mut i = 1;
                while i + 7 < nx - 1 {
                    let idx = i + j * nx + k * nx * ny;

                    let p_curr = _mm256_loadu_ps(pressure.as_ptr().add(idx));
                    let p_prev = _mm256_loadu_ps(pressure_prev.as_ptr().add(idx));
                    let lap = _mm256_loadu_ps(laplacian.as_ptr().add(idx));

                    let temp = _mm256_mul_ps(two, p_curr);
                    let temp = _mm256_sub_ps(temp, p_prev);
                    let temp = _mm256_fmadd_ps(c_dt2, lap, temp);

                    _mm256_storeu_ps(pressure.as_mut_ptr().add(idx), temp);
                    i += 8;
                }

                // Scalar tail
                while i < nx - 1 {
                    let idx = i + j * nx + k * nx * ny;
                    pressure[idx] = c_squared_dt_squared.mul_add(
                        laplacian[idx],
                        2.0f32.mul_add(pressure[idx], -pressure_prev[idx]),
                    );
                    i += 1;
                }
            }
        }
    }

    /// AVX-512 pressure update (16-wide f32 lanes).
    ///
    /// Evaluates p^{n+1} = 2p^n - p^{n-1} + c²Δt² · Lap(p^n).
    ///
    /// # Safety
    ///
    /// - CPU detection ensures AVX-512F support.
    /// - Loop guard `i + 15 < nx - 1` keeps 16-lane loads in the row interior.
    /// - `_mm512_loadu_ps` / `_mm512_storeu_ps` allow unaligned addresses.
    ///
    /// **Preconditions**: all slices have length ≥ nx · ny · nz.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    #[allow(unsafe_code)]
    pub(super) unsafe fn update_pressure_avx512(
        &self,
        pressure: &mut [f32],
        pressure_prev: &[f32],
        laplacian: &[f32],
        c_squared_dt_squared: f32,
        nx: usize,
        ny: usize,
        nz: usize,
    ) {
        use std::arch::x86_64::{
            _mm512_add_ps, _mm512_loadu_ps, _mm512_mul_ps, _mm512_set1_ps, _mm512_storeu_ps,
            _mm512_sub_ps,
        };

        let two = _mm512_set1_ps(2.0);
        let c_dt2 = _mm512_set1_ps(c_squared_dt_squared);

        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                let mut i = 1;
                while i + 15 < nx - 1 {
                    let idx = i + j * nx + k * nx * ny;

                    let p_curr = _mm512_loadu_ps(pressure.as_ptr().add(idx));
                    let p_prev = _mm512_loadu_ps(pressure_prev.as_ptr().add(idx));
                    let lap = _mm512_loadu_ps(laplacian.as_ptr().add(idx));

                    let doubled = _mm512_mul_ps(two, p_curr);
                    let inertial = _mm512_sub_ps(doubled, p_prev);
                    let forcing = _mm512_mul_ps(c_dt2, lap);
                    let next = _mm512_add_ps(inertial, forcing);

                    _mm512_storeu_ps(pressure.as_mut_ptr().add(idx), next);
                    i += 16;
                }

                // Scalar tail
                while i < nx - 1 {
                    let idx = i + j * nx + k * nx * ny;
                    pressure[idx] = c_squared_dt_squared.mul_add(
                        laplacian[idx],
                        2.0f32.mul_add(pressure[idx], -pressure_prev[idx]),
                    );
                    i += 1;
                }
            }
        }
    }
}
