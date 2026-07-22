//! AVX-512 velocity-field stencil update.
//!
//! SRP: changes when the velocity momentum equation, gradient stencil order, or
//! boundary exclusion policy changes.
//!
//! ## Discretisation (Euler momentum, component `dim`)
//!
//! ```text
//! u^(n+1)`i` = u^n`i` − (Δt/(ρΔx)) × (p[i+1] − p[i-1]) / 2
//! ```
//!
//! Vectorised form (8-wide AVX-512, f64):
//! ```text
//! u_new = u + coeff × (p_plus − p_minus)
//! ```

use super::{FdtdAvx512StencilProcessor, AVX512_F64_LANES};
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3;

impl FdtdAvx512StencilProcessor {
    /// Update velocity field with AVX-512 acceleration.
    ///
    /// # Arguments
    /// * `u`   — velocity component field to update in-place
    /// * `p`   — current pressure field
    /// * `dim` — spatial dimension: 0 (x), 1 (y), 2 (z)
    /// # Errors
    /// - Returns [`crate::KwaversError::FeatureNotAvailable`] if the precondition for a FeatureNotAvailable-class constraint is violated.
    /// - Returns [`crate::KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn update_velocity_avx512(
        &self,
        u: &mut Array3<f64>,
        p: &Array3<f64>,
        dim: usize,
    ) -> KwaversResult<()> {
        if p.shape() != [self.nx, self.ny, self.nz] || u.shape() != p.shape() {
            return Err(KwaversError::InvalidInput(
                "Velocity and pressure fields must match processor dimensions".to_owned(),
            ));
        }
        if !p.layout().is_c_contiguous() || !u.layout().is_c_contiguous() {
            return Err(KwaversError::InvalidInput(
                "AVX-512 velocity fields must use C-contiguous Leto layouts".to_owned(),
            ));
        }
        if dim > 2 {
            return Err(KwaversError::InvalidInput(
                "Velocity dimension must be 0 (x), 1 (y), or 2 (z)".to_owned(),
            ));
        }

        #[cfg(target_arch = "x86_64")]
        {
            if !is_x86_feature_detected!("avx512f") {
                return Err(KwaversError::FeatureNotAvailable(
                    "AVX-512F not detected at runtime".to_owned(),
                ));
            }

            // SAFETY: dimensions, C-contiguous layouts, and dim are validated above;
            // AVX-512F is detected immediately before this target-feature call.
            #[allow(unsafe_code)]
            unsafe {
                self.update_velocity_avx512_unsafe(u, p, dim)?;
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        return Err(KwaversError::FeatureNotAvailable(
            "AVX-512 velocity update only available on x86_64".to_string(),
        ));

        Ok(())
    }

    /// Unsafe AVX-512 velocity kernel.
    ///
    /// # Safety
    ///
    /// Preconditions (all verified by `update_velocity_avx512` before calling):
    /// 1. `u.shape() == p.shape() == (self.nx, self.ny, self.nz)`.
    /// 2. `dim ∈ {0, 1, 2}`.
    /// 3. AVX-512F is available (checked immediately before the call).
    /// 4. Both arrays are standard-layout (C-order, contiguous).
    ///
    /// Loop bounds guarantee all pointer offsets ±stride remain within the
    /// allocated region (same argument as for the pressure kernel).
    /// `u_ptr` is an exclusive mutable pointer; `p_ptr` is immutable read-only.
    /// # Errors
    /// - Returns [`crate::KwaversError::FeatureNotAvailable`] if the precondition for a FeatureNotAvailable-class constraint is violated.
    ///
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[allow(unsafe_code)]
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn update_velocity_avx512_unsafe(
        &self,
        u: &mut Array3<f64>,
        p: &Array3<f64>,
        dim: usize,
    ) -> KwaversResult<()> {
        use std::arch::x86_64::{
            _mm512_add_pd, _mm512_loadu_pd, _mm512_mul_pd, _mm512_set1_pd, _mm512_storeu_pd,
            _mm512_sub_pd,
        };

        let p_ptr = p.as_ptr();
        let u_ptr = u
            .as_slice_memory_order_mut()
            .expect("invariant: AVX-512 velocity output field must be contiguous")
            .as_mut_ptr();
        let coeff_vec = _mm512_set1_pd(self.velocity_coeff);

        let row_stride = self.nz;
        let plane_stride = self.ny * row_stride;

        // Choose stencil stride based on spatial dimension.
        // Leto C order makes z unit-stride: x=ny*nz, y=nz, z=1.
        let stride = match dim {
            0 => plane_stride,
            1 => row_stride,
            2 => 1,
            _ => unreachable!(),
        };

        for x in 1..self.nx - 1 {
            for y in 1..self.ny - 1 {
                let row_base = x * plane_stride + y * row_stride;
                let mut z = 1;

                while z + AVX512_F64_LANES < self.nz {
                    let idx = row_base + z;
                    let p_plus = _mm512_loadu_pd(p_ptr.add(idx + stride));
                    let p_minus = _mm512_loadu_pd(p_ptr.add(idx - stride));
                    let grad = _mm512_sub_pd(p_plus, p_minus);
                    let u_update = _mm512_mul_pd(coeff_vec, grad);
                    let u_val = _mm512_loadu_pd(u_ptr.add(idx));
                    let u_new = _mm512_add_pd(u_val, u_update);
                    _mm512_storeu_pd(u_ptr.add(idx), u_new);
                    z += AVX512_F64_LANES;
                }

                for z in z..self.nz - 1 {
                    let idx = row_base + z;
                    let gradient = *p_ptr.add(idx + stride) - *p_ptr.add(idx - stride);
                    *u_ptr.add(idx) += self.velocity_coeff * gradient;
                }
            }
        }

        Ok(())
    }
}
