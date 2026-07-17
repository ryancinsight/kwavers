//! AVX-512 pressure-field stencil update.
//!
//! SRP: changes when the pressure leapfrog stencil, tiling strategy, or boundary-condition
//! policy changes.
//!
//! ## Mathematical Model (3-D Acoustic Wave Equation)
//!
//! ```text
//! p^(n+1)[i,j,k] = (2 - c²Δt²/Δx² × 6) × p^n[i,j,k]
//!                  - p^(n-1)[i,j,k]
//!                  + c²Δt²/Δx² × (p^n[i-1,j,k] + p^n[i+1,j,k]
//!                                 + p^n[i,j-1,k] + p^n[i,j+1,k]
//!                                 + p^n[i,j,k-1] + p^n[i,j,k+1])
//! ```
//!
//! Vectorised form (8-wide AVX-512, f64):
//! ```text
//! v_p_new = coeff_central×v_p + (coeff×laplacian − v_p_prev)
//! ```

use super::{FdtdAvx512StencilProcessor, AVX512_F64_LANES};
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3;

impl FdtdAvx512StencilProcessor {
    /// Update pressure field with AVX-512 acceleration.
    ///
    /// Processes interior data in 8×tile×tile spatial tiles to maximise L1 cache
    /// utilisation. The eight-wide vector lane follows Leto's unit-stride final
    /// axis; scalar tails preserve every interior cell. The zero-initialised output
    /// provides Dirichlet boundary conditions on all six faces.
    ///
    /// # Arguments
    /// * `p_curr` — current pressure field (time step n)
    /// * `p_prev` — previous pressure field (time step n-1)
    /// * `u_div`  — divergence of velocity field (currently unused; reserved for source terms)
    ///
    /// # Returns
    /// Updated pressure field at time step n+1.
    /// # Errors
    /// - Returns [`crate::KwaversError::FeatureNotAvailable`] if the precondition for a FeatureNotAvailable-class constraint is violated.
    /// - Returns [`crate::KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn update_pressure_avx512(
        &self,
        p_curr: &Array3<f64>,
        p_prev: &Array3<f64>,
        u_div: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        if p_curr.shape() != p_prev.shape() || p_curr.shape() != u_div.shape() {
            return Err(KwaversError::InvalidInput(
                "All fields must have identical dimensions".to_owned(),
            ));
        }
        let shape = p_curr.shape();
        if shape != [self.nx, self.ny, self.nz] {
            return Err(KwaversError::InvalidInput(
                "Field dimensions do not match processor configuration".to_owned(),
            ));
        }
        if !p_curr.layout().is_c_contiguous()
            || !p_prev.layout().is_c_contiguous()
            || !u_div.layout().is_c_contiguous()
        {
            return Err(KwaversError::InvalidInput(
                "AVX-512 pressure fields must use C-contiguous Leto layouts".to_owned(),
            ));
        }

        let mut p_new = Array3::zeros(shape);

        #[cfg(target_arch = "x86_64")]
        {
            if !is_x86_feature_detected!("avx512f") {
                return Err(KwaversError::FeatureNotAvailable(
                    "AVX-512F not detected at runtime".to_owned(),
                ));
            }

            // SAFETY: field dimensions and C-contiguous layouts are validated above;
            // AVX-512F is detected immediately before this target-feature call.
            #[allow(unsafe_code)]
            unsafe {
                self.update_pressure_avx512_unsafe(p_curr, p_prev, u_div, &mut p_new)?;
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        return Err(KwaversError::FeatureNotAvailable(
            "AVX-512 stencil only available on x86_64".to_string(),
        ));

        Ok(p_new)
    }

    /// Unsafe AVX-512 pressure kernel.
    ///
    /// # Safety
    ///
    /// Preconditions (all verified by `update_pressure_avx512` before calling):
    /// 1. `p_curr.shape() == p_prev.shape() == u_div.shape() == p_new.shape() == (self.nx, self.ny, self.nz)`
    /// 2. `(self.nx, self.ny, self.nz) >= (4, 4, 4)` (enforced by constructor).
    /// 3. AVX-512F is available (checked immediately before the call).
    /// 4. All arrays are standard-layout (C-order, row-major, contiguous allocation).
    ///
    /// Loop bounds guarantee all pointer offsets ± (1, nz, ny×nz) remain within the
    /// allocated region:
    /// - Interior vector lanes satisfy `z+7 < nz−1`; scalar tails cover the
    ///   remaining `z ∈ [1, nz−1)` cells.
    /// - Neighbor offsets ±1, ±nz, ±(ny×nz) are bounded analogously.
    ///
    /// `p_new` is exclusively owned (no aliasing); `p_curr`/`p_prev`/`u_div` are immutable.
    /// # Errors
    /// - Returns [`crate::KwaversError::FeatureNotAvailable`] if the precondition for a FeatureNotAvailable-class constraint is violated.
    ///
    #[allow(unsafe_code)]
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn update_pressure_avx512_unsafe(
        &self,
        p_curr: &Array3<f64>,
        p_prev: &Array3<f64>,
        u_div: &Array3<f64>,
        p_new: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        use std::arch::x86_64::{
            _mm512_add_pd, _mm512_fmadd_pd, _mm512_loadu_pd, _mm512_mul_pd, _mm512_set1_pd,
            _mm512_storeu_pd, _mm512_sub_pd,
        };

        if !is_x86_feature_detected!("avx512f") {
            return Err(KwaversError::FeatureNotAvailable(
                "AVX-512F not detected at runtime".to_owned(),
            ));
        }

        let p_curr_ptr = p_curr.as_ptr();
        let p_prev_ptr = p_prev.as_ptr();
        let _ = u_div;
        let p_new_ptr = p_new
            .as_slice_memory_order_mut()
            .expect("invariant: AVX-512 stencil output field must be contiguous")
            .as_mut_ptr();

        let coeff_central = _mm512_set1_pd(self.pressure_central_coeff);
        let coeff = _mm512_set1_pd(self.pressure_coeff);

        let tile_size = self.config.tile_size.min(AVX512_F64_LANES);
        let row_stride = self.nz;
        let plane_stride = self.ny * row_stride;

        // Array3 uses C order: `[x, y, z]` maps to
        // `x * (ny * nz) + y * nz + z`, so `z` is the vector-contiguous axis.
        for x_tile in (1..self.nx - 1).step_by(tile_size) {
            let x_end = (x_tile + tile_size).min(self.nx - 1);
            for y_tile in (1..self.ny - 1).step_by(tile_size) {
                let y_end = (y_tile + tile_size).min(self.ny - 1);
                for x in x_tile..x_end {
                    for y in y_tile..y_end {
                        let row_base = x * plane_stride + y * row_stride;
                        let mut z = 1;

                        while z + AVX512_F64_LANES < self.nz {
                            let idx = row_base + z;
                            let p_curr_vec = _mm512_loadu_pd(p_curr_ptr.add(idx));
                            let p_prev_vec = _mm512_loadu_pd(p_prev_ptr.add(idx));
                            let p_x_minus = _mm512_loadu_pd(p_curr_ptr.add(idx - plane_stride));
                            let p_x_plus = _mm512_loadu_pd(p_curr_ptr.add(idx + plane_stride));
                            let p_y_minus = _mm512_loadu_pd(p_curr_ptr.add(idx - row_stride));
                            let p_y_plus = _mm512_loadu_pd(p_curr_ptr.add(idx + row_stride));
                            let p_z_minus = _mm512_loadu_pd(p_curr_ptr.add(idx - 1));
                            let p_z_plus = _mm512_loadu_pd(p_curr_ptr.add(idx + 1));

                            let mut neighbor_sum = _mm512_add_pd(p_x_minus, p_x_plus);
                            neighbor_sum = _mm512_add_pd(neighbor_sum, p_y_minus);
                            neighbor_sum = _mm512_add_pd(neighbor_sum, p_y_plus);
                            neighbor_sum = _mm512_add_pd(neighbor_sum, p_z_minus);
                            neighbor_sum = _mm512_add_pd(neighbor_sum, p_z_plus);

                            let p_new_vec = _mm512_fmadd_pd(
                                coeff_central,
                                p_curr_vec,
                                _mm512_sub_pd(_mm512_mul_pd(coeff, neighbor_sum), p_prev_vec),
                            );
                            _mm512_storeu_pd(p_new_ptr.add(idx), p_new_vec);
                            z += AVX512_F64_LANES;
                        }

                        for z in z..self.nz - 1 {
                            let idx = row_base + z;
                            let neighbor_sum = *p_curr_ptr.add(idx - plane_stride)
                                + *p_curr_ptr.add(idx + plane_stride)
                                + *p_curr_ptr.add(idx - row_stride)
                                + *p_curr_ptr.add(idx + row_stride)
                                + *p_curr_ptr.add(idx - 1)
                                + *p_curr_ptr.add(idx + 1);
                            *p_new_ptr.add(idx) = self.pressure_central_coeff.mul_add(
                                *p_curr_ptr.add(idx),
                                self.pressure_coeff
                                    .mul_add(neighbor_sum, -*p_prev_ptr.add(idx)),
                            );
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
