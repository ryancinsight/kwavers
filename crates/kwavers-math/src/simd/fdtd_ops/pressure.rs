//! Pressure update kernel backed by `hermes-simd` runtime dispatch.
//!
//! Implements the second-order accurate leapfrog time step
//!
//! ```text
//! p^{n+1} = 2·p^n − p^{n−1} + c²Δt²·∇²p^n
//! ```
//!
//! on the interior points of a 3-D grid using `hermes_simd::scale` and
//! `hermes_simd::axpy`, which select AVX-512 / AVX2 / NEON / scalar at
//! runtime.  All unsafe intrinsics are in `hermes_simd_intrinsics`.

use super::FdtdSimdOps;
use hermes_simd::{axpy, scale};

impl FdtdSimdOps {
    /// Hermes-dispatched 3-D FDTD pressure update.
    ///
    /// Processes interior rows (i ∈ 1..nx−1, j ∈ 1..ny−1, k ∈ 1..nz−1)
    /// via `hermes_simd` kernels to avoid hand-rolled intrinsics.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn update_pressure_hermes(
        &self,
        pressure: &mut [f32],
        pressure_prev: &[f32],
        laplacian: &[f32],
        c_squared_dt_squared: f32,
        nx: usize,
        ny: usize,
        nz: usize,
    ) {
        let row_len = nx - 2;
        let mut temp = vec![0.0f32; row_len];

        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                let row_start = 1 + j * nx + k * nx * ny;
                let row_end = row_start + row_len;

                // temp = p^n[row]
                temp.copy_from_slice(&pressure[row_start..row_end]);

                // temp = 2·p^n
                scale(&mut temp, 2.0f32);

                // temp = 2·p^n − p^{n−1}
                axpy(-1.0f32, &pressure_prev[row_start..row_end], &mut temp)
                    .expect("pressure axpy(-1, p_prev): length mismatch");

                // temp = 2·p^n − p^{n−1} + c²Δt²·∇²p^n
                axpy(
                    c_squared_dt_squared,
                    &laplacian[row_start..row_end],
                    &mut temp,
                )
                .expect("pressure axpy(c²Δt², lap): length mismatch");

                pressure[row_start..row_end].copy_from_slice(&temp);
            }
        }
    }
}
