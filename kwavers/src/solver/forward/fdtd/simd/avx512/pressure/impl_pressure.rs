//! AVX-512 pressure field update implementation for `Avx512StencilProcessor`.

use super::super::Avx512StencilProcessor;
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;

impl Avx512StencilProcessor {
    /// Update pressure field with AVX-512 acceleration.
    ///
    /// Implements 3D acoustic FDTD pressure update with 8-wide vectorization.
    /// Processes data in tiles to maximize cache utilization.
    ///
    /// # Arguments
    /// * `p_curr` — Current pressure field (time step n)
    /// * `p_prev` — Previous pressure field (time step n-1)
    /// * `u_div` — Divergence of velocity field
    ///
    /// # Returns
    /// Updated pressure field at time step n+1
    pub fn update_pressure_avx512(
        &self,
        p_curr: &Array3<f64>,
        p_prev: &Array3<f64>,
        u_div: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        if p_curr.shape() != p_prev.shape() || p_curr.shape() != u_div.shape() {
            return Err(KwaversError::InvalidInput(
                "All fields must have identical dimensions".to_string(),
            ));
        }

        let shape = p_curr.dim();
        if shape != (self.nx, self.ny, self.nz) {
            return Err(KwaversError::InvalidInput(
                "Field dimensions do not match processor configuration".to_string(),
            ));
        }

        let mut p_new = Array3::zeros(shape);

        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: All safety requirements documented in update_pressure_avx512_unsafe:
            // - Arrays validated to have matching dimensions
            // - AVX-512F support checked at runtime
            #[allow(unsafe_code)]
            unsafe {
                self.update_pressure_avx512_unsafe(p_curr, p_prev, u_div, &mut p_new)?;
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            return Err(KwaversError::FeatureNotAvailable(
                "AVX-512 stencil only available on x86_64".to_string(),
            ));
        }

        Ok(p_new)
    }

    /// Unsafe AVX-512 implementation (internal).
    ///
    /// # Safety
    ///
    /// This function is unsafe because it:
    /// 1. Uses raw pointer arithmetic to access array elements
    /// 2. Calls intrinsic functions from `std::arch::x86_64` that require:
    ///    - AVX-512F CPU feature support (verified at runtime)
    ///    - Proper memory alignment (guaranteed by ndarray's allocation)
    ///    - Valid pointer offsets within array bounds (validated by loop bounds)
    ///
    /// Caller must ensure:
    /// - Input arrays have matching dimensions equal to `(self.nx, self.ny, self.nz)`
    /// - Arrays are properly initialized (no uninitialized memory)
    #[allow(unsafe_code)]
    #[cfg(target_arch = "x86_64")]
    unsafe fn update_pressure_avx512_unsafe(
        &self,
        p_curr: &Array3<f64>,
        p_prev: &Array3<f64>,
        u_div: &Array3<f64>,
        p_new: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        use std::arch::x86_64::*;

        if !is_x86_feature_detected!("avx512f") {
            return Err(KwaversError::FeatureNotAvailable(
                "AVX-512F not detected at runtime".to_string(),
            ));
        }

        // SAFETY: Extract raw pointers for AVX-512 intrinsics.
        // - p_curr, p_prev, u_div: immutable borrows → read-only pointers valid for array lifetime
        // - p_new: mutable exclusive borrow → writable pointer, no aliasing
        // - Dimension validation: Public API ensures all arrays have shape (nx, ny, nz)
        // - Memory layout: ndarray guarantees contiguous C-order (row-major) allocation
        let p_curr_ptr = p_curr.as_ptr();
        let p_prev_ptr = p_prev.as_ptr();
        let _u_div_ptr = u_div.as_ptr();
        let p_new_ptr = p_new.as_mut_ptr();

        let coeff_central = _mm512_set1_pd(self.pressure_central_coeff);
        let coeff = _mm512_set1_pd(self.pressure_coeff);
        let two = _mm512_set1_pd(2.0);

        let tile_size = self.config.tile_size.min(8);
        let stride_xy = self.ny as isize;
        let stride_z = (self.nx * self.ny) as isize;

        for z_tile in (1..self.nz - 1).step_by(tile_size) {
            for y_tile in (1..self.ny - 1).step_by(tile_size) {
                for x_base in (1..self.nx - 1).step_by(8) {
                    let x = x_base;
                    let y = y_tile;
                    let z = z_tile;
                    let idx = (z * self.nx * self.ny + y * self.nx + x) as isize;

                    // SAFETY: AVX-512 vectorized load of 8 consecutive f64 pressure values.
                    // Bounds: idx + 7 < total_size (enforced by loop step and interior restriction).
                    let p_curr_vec = _mm512_loadu_pd(p_curr_ptr.offset(idx));
                    let p_prev_vec = _mm512_loadu_pd(p_prev_ptr.offset(idx));

                    // SAFETY: x-direction neighbors (stride = 1). Bounds: idx ± 1 in-bounds by loop invariants.
                    let p_x_minus = _mm512_loadu_pd(p_curr_ptr.offset(idx - 1));
                    let p_x_plus = _mm512_loadu_pd(p_curr_ptr.offset(idx + 1));

                    // SAFETY: y-direction neighbors (stride = nx). Bounds: idx ± nx in-bounds by loop invariants.
                    let p_y_minus = _mm512_loadu_pd(p_curr_ptr.offset(idx - stride_xy));
                    let p_y_plus = _mm512_loadu_pd(p_curr_ptr.offset(idx + stride_xy));

                    // SAFETY: z-direction neighbors (stride = nx×ny). Bounds: idx ± nx×ny in-bounds by loop invariants.
                    let p_z_minus = _mm512_loadu_pd(p_curr_ptr.offset(idx - stride_z));
                    let p_z_plus = _mm512_loadu_pd(p_curr_ptr.offset(idx + stride_z));

                    // 7-point Laplacian: Σ(6 neighbors)
                    let mut laplacian = _mm512_add_pd(p_x_minus, p_x_plus);
                    laplacian = _mm512_add_pd(laplacian, p_y_minus);
                    laplacian = _mm512_add_pd(laplacian, p_y_plus);
                    laplacian = _mm512_add_pd(laplacian, p_z_minus);
                    laplacian = _mm512_add_pd(laplacian, p_z_plus);

                    let laplacian_term = _mm512_mul_pd(coeff, laplacian);

                    // Leapfrog update: p^(n+1) = coeff_central * p^n + (laplacian_term - p^(n-1))
                    let _two_p_curr = _mm512_mul_pd(two, p_curr_vec);
                    let p_new_vec = _mm512_fmadd_pd(
                        coeff_central,
                        p_curr_vec,
                        _mm512_sub_pd(laplacian_term, p_prev_vec),
                    );

                    // SAFETY: Store 8 computed values. p_new_ptr is exclusive mutable, indices in-bounds.
                    _mm512_storeu_pd(p_new_ptr.offset(idx), p_new_vec);
                }
            }
        }

        // Apply Dirichlet boundary conditions (zero pressure on all 6 faces).
        // SAFETY: All offsets computed from valid loop ranges [0, n).
        for i in 0..self.nx {
            for j in 0..self.ny {
                *p_new_ptr.add(j * self.nx + i) = 0.0;
                *p_new_ptr.add((self.nz - 1) * self.nx * self.ny + j * self.nx + i) = 0.0;
            }
        }

        for i in 0..self.nx {
            for k in 0..self.nz {
                *p_new_ptr.add(k * self.nx * self.ny + i) = 0.0;
                *p_new_ptr.add(k * self.nx * self.ny + (self.ny - 1) * self.nx + i) = 0.0;
            }
        }

        for j in 0..self.ny {
            for k in 0..self.nz {
                *p_new_ptr.add(k * self.nx * self.ny + j * self.nx) = 0.0;
                *p_new_ptr.add(k * self.nx * self.ny + j * self.nx + self.nx - 1) = 0.0;
            }
        }

        Ok(())
    }
}
