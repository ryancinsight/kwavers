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

use super::FdtdAvx512StencilProcessor;
use kwavers_core::error::{KwaversError, KwaversResult};
use ndarray::Array3;

impl FdtdAvx512StencilProcessor {
    /// Update pressure field with AVX-512 acceleration.
    ///
    /// Processes interior data in 8×tile×tile spatial tiles to maximise L1 cache
    /// utilisation. Dirichlet (zero) boundary conditions are applied on all six faces.
    ///
    /// # Arguments
    /// * `p_curr` — current pressure field (time step n)
    /// * `p_prev` — previous pressure field (time step n-1)
    /// * `u_div`  — divergence of velocity field (currently unused; reserved for source terms)
    ///
    /// # Returns
    /// Updated pressure field at time step n+1.
    /// # Errors
    /// - Returns [`KwaversError::FeatureNotAvailable`] if the precondition for a FeatureNotAvailable-class constraint is violated.
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
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
        let shape = p_curr.dim();
        if shape != (self.nx, self.ny, self.nz) {
            return Err(KwaversError::InvalidInput(
                "Field dimensions do not match processor configuration".to_owned(),
            ));
        }

        let mut p_new = Array3::zeros(shape);

        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: dimension-match and AVX-512 availability validated above.
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
    /// 1. `p_curr.dim() == p_prev.dim() == u_div.dim() == p_new.dim() == (self.nx, self.ny, self.nz)`
    /// 2. `(self.nx, self.ny, self.nz) >= (4, 4, 4)` (enforced by constructor).
    /// 3. AVX-512F is available (checked at construction time on x86_64).
    /// 4. All arrays are standard-layout (C-order, row-major, contiguous allocation).
    ///
    /// Loop bounds guarantee all pointer offsets ± (1, nx, nx×ny) remain within the
    /// allocated region:
    /// - Interior loop: `x ∈ [1, nx−1)` step 8 ⇒ `idx+7 < nx×ny×nz`.
    /// - Neighbor offsets ±1, ±nx, ±(nx×ny) are bounded analogously.
    ///
    /// `p_new` is exclusively owned (no aliasing); `p_curr`/`p_prev`/`u_div` are immutable.
    /// # Errors
    /// - Returns [`KwaversError::FeatureNotAvailable`] if the precondition for a FeatureNotAvailable-class constraint is violated.
    ///
    #[allow(unsafe_code)]
    #[cfg(target_arch = "x86_64")]
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
        let _u_div_ptr = u_div.as_ptr();
        let p_new_ptr = p_new.as_mut_ptr();

        let coeff_central = _mm512_set1_pd(self.pressure_central_coeff);
        let coeff = _mm512_set1_pd(self.pressure_coeff);

        let tile_size = self.config.tile_size.min(8);
        let stride_xy = self.ny as isize;
        let stride_z = (self.nx * self.ny) as isize;

        // Interior update — 8-wide AVX-512 leapfrog stencil.
        // Loop bounds ensure all pointer offsets stay within the allocated region
        // (see Safety contract above).
        for z_tile in (1..self.nz - 1).step_by(tile_size) {
            for y_tile in (1..self.ny - 1).step_by(tile_size) {
                for x_base in (1..self.nx - 1).step_by(8) {
                    let idx = (z_tile * self.nx * self.ny + y_tile * self.nx + x_base) as isize;

                    let p_curr_vec = _mm512_loadu_pd(p_curr_ptr.offset(idx));
                    let p_prev_vec = _mm512_loadu_pd(p_prev_ptr.offset(idx));

                    // x-neighbors (stride 1)
                    let p_x_minus = _mm512_loadu_pd(p_curr_ptr.offset(idx - 1));
                    let p_x_plus = _mm512_loadu_pd(p_curr_ptr.offset(idx + 1));
                    // y-neighbors (stride nx)
                    let p_y_minus = _mm512_loadu_pd(p_curr_ptr.offset(idx - stride_xy));
                    let p_y_plus = _mm512_loadu_pd(p_curr_ptr.offset(idx + stride_xy));
                    // z-neighbors (stride nx×ny)
                    let p_z_minus = _mm512_loadu_pd(p_curr_ptr.offset(idx - stride_z));
                    let p_z_plus = _mm512_loadu_pd(p_curr_ptr.offset(idx + stride_z));

                    // Laplacian = Σ six neighbors
                    let mut laplacian = _mm512_add_pd(p_x_minus, p_x_plus);
                    laplacian = _mm512_add_pd(laplacian, p_y_minus);
                    laplacian = _mm512_add_pd(laplacian, p_y_plus);
                    laplacian = _mm512_add_pd(laplacian, p_z_minus);
                    laplacian = _mm512_add_pd(laplacian, p_z_plus);

                    let laplacian_term = _mm512_mul_pd(coeff, laplacian);

                    // Leapfrog: p^(n+1) = coeff_central×p^n + (coeff×∇²p^n − p^(n-1))
                    let p_new_vec = _mm512_fmadd_pd(
                        coeff_central,
                        p_curr_vec,
                        _mm512_sub_pd(laplacian_term, p_prev_vec),
                    );

                    _mm512_storeu_pd(p_new_ptr.offset(idx), p_new_vec);
                }
            }
        }

        // Dirichlet (zero) boundary conditions on all six faces.
        // Boundary write ranges are disjoint from the interior loop region.
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
