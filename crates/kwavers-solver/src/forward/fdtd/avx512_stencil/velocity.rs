//! AVX-512 velocity-field stencil update.
//!
//! SRP: changes when the velocity momentum equation, gradient stencil order, or
//! boundary exclusion policy changes.
//!
//! ## Discretisation (Euler momentum, component `dim`)
//!
//! ```text
//! u^(n+1)[i] = u^n[i] − (Δt/(ρΔx)) × (p[i+1] − p[i-1]) / 2
//! ```
//!
//! Vectorised form (8-wide AVX-512, f64):
//! ```text
//! u_new = u + coeff × (p_plus − p_minus)
//! ```

use super::FdtdAvx512StencilProcessor;
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
    /// - Returns [`KwaversError::FeatureNotAvailable`] if the precondition for a FeatureNotAvailable-class constraint is violated.
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn update_velocity_avx512(
        &self,
        u: &mut Array3<f64>,
        p: &Array3<f64>,
        dim: usize,
    ) -> KwaversResult<()> {
        if p.shape() != (self.nx, self.ny, self.nz) {
            return Err(KwaversError::InvalidInput(
                "Pressure field dimensions mismatch".to_owned(),
            ));
        }
        if dim > 2 {
            return Err(KwaversError::InvalidInput(
                "Velocity dimension must be 0 (x), 1 (y), or 2 (z)".to_owned(),
            ));
        }

        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: dimension-match and dim-range validated above.
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
    /// 3. AVX-512F is available.
    /// 4. Both arrays are standard-layout (C-order, contiguous).
    ///
    /// Loop bounds guarantee all pointer offsets ±stride remain within the
    /// allocated region (same argument as for the pressure kernel).
    /// `u_ptr` is an exclusive mutable pointer; `p_ptr` is immutable read-only.
    /// # Errors
    /// - Returns [`KwaversError::FeatureNotAvailable`] if the precondition for a FeatureNotAvailable-class constraint is violated.
    ///
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[allow(unsafe_code)]
    #[cfg(target_arch = "x86_64")]
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

        if !is_x86_feature_detected!("avx512f") {
            return Err(KwaversError::FeatureNotAvailable(
                "AVX-512F not available".to_owned(),
            ));
        }

        let p_ptr = p.as_ptr();
        let u_ptr = u.as_mut_ptr();
        let coeff_vec = _mm512_set1_pd(self.velocity_coeff);

        let stride_xy = self.ny as isize;
        let stride_z = (self.nx * self.ny) as isize;

        // Choose stencil stride based on spatial dimension.
        // dim=0 → stride=1 (x), dim=1 → stride=nx (y), dim=2 → stride=nx×ny (z).
        let stride: isize = match dim {
            0 => 1,
            1 => stride_xy,
            2 => stride_z,
            _ => unreachable!(),
        };

        // Interior update — 8-wide AVX-512 central-difference momentum equation.
        // Loop bounds bound all ±stride pointer offsets within the allocated region.
        for z in 1..self.nz - 1 {
            for y in 1..self.ny - 1 {
                for x in (1..self.nx - 1).step_by(8) {
                    let idx = (z * self.nx * self.ny + y * self.nx + x) as isize;
                    let p_plus = _mm512_loadu_pd(p_ptr.offset(idx + stride));
                    let p_minus = _mm512_loadu_pd(p_ptr.offset(idx - stride));
                    let grad = _mm512_sub_pd(p_plus, p_minus);
                    let u_update = _mm512_mul_pd(coeff_vec, grad);
                    let u_val = _mm512_loadu_pd(u_ptr.offset(idx));
                    let u_new = _mm512_add_pd(u_val, u_update);
                    _mm512_storeu_pd(u_ptr.offset(idx), u_new);
                }
            }
        }

        Ok(())
    }
}
