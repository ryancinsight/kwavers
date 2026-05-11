//! WENO3 (third-order) shock limiting implementation.
//!
//! Jiang-Shu smoothness indicators (J. Comput. Phys. 126:202-228, 1996).

use super::types::WENOLimiter;
use crate::core::constants::numerical::{
    STENCIL_COEFF_1_4, WENO_WEIGHT_0, WENO_WEIGHT_1, WENO_WEIGHT_2,
};
use crate::core::error::KwaversResult;
use ndarray::Array3;

impl WENOLimiter {
    /// WENO3 limiting — zero-allocation version.
    ///
    /// Reads stencil values from the immutable `src` (the pre-limited field) and
    /// writes limited values into `output`. `output` must be pre-initialized to
    /// `src` so that unshocked cells keep their original values.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn weno3_limit_into(
        &self,
        src: &Array3<f64>,
        shock_indicator: &Array3<f64>,
        output: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = src.dim();
        for i in 2..nx - 2 {
            for j in 2..ny - 2 {
                for k in 2..nz - 2 {
                    if shock_indicator[[i, j, k]] > 0.5 {
                        // Read stencil from immutable src — no write-read conflict.
                        let weno_x = self.weno3_stencil(&[
                            src[[i - 2, j, k]],
                            src[[i - 1, j, k]],
                            src[[i, j, k]],
                            src[[i + 1, j, k]],
                            src[[i + 2, j, k]],
                        ]);
                        let weno_y = self.weno3_stencil(&[
                            src[[i, j - 2, k]],
                            src[[i, j - 1, k]],
                            src[[i, j, k]],
                            src[[i, j + 1, k]],
                            src[[i, j + 2, k]],
                        ]);
                        let weno_z = self.weno3_stencil(&[
                            src[[i, j, k - 2]],
                            src[[i, j, k - 1]],
                            src[[i, j, k]],
                            src[[i, j, k + 1]],
                            src[[i, j, k + 2]],
                        ]);
                        output[[i, j, k]] = (weno_x + weno_y + weno_z) / 3.0;
                    }
                }
            }
        }
        Ok(())
    }

    /// WENO3 stencil computation
    pub(super) fn weno3_stencil(&self, v: &[f64; 5]) -> f64 {
        // Three candidate stencils
        let q0 = v[0] / 3.0 - 7.0 * v[1] / 6.0 + 11.0 * v[2] / 6.0;
        let q1 = -v[1] / 6.0 + 5.0 * v[2] / 6.0 + v[3] / 3.0;
        let q2 = v[2] / 3.0 + 5.0 * v[3] / 6.0 - v[4] / 6.0;

        // Smoothness indicators (Jiang-Shu)
        let beta0 = (13.0_f64 / 12.0).mul_add((2.0f64.mul_add(-v[1], v[0]) + v[2]).powi(2), STENCIL_COEFF_1_4 * 3.0f64.mul_add(v[2], 4.0f64.mul_add(-v[1], v[0])).powi(2));
        let beta1 = (13.0_f64 / 12.0).mul_add((2.0f64.mul_add(-v[2], v[1]) + v[3]).powi(2), STENCIL_COEFF_1_4 * (v[1] - v[3]).powi(2));
        let beta2 = (13.0_f64 / 12.0).mul_add((2.0f64.mul_add(-v[3], v[2]) + v[4]).powi(2), STENCIL_COEFF_1_4 * (3.0f64.mul_add(v[2], -(4.0 * v[3])) + v[4]).powi(2));

        // Optimal weights
        let d0 = WENO_WEIGHT_0;
        let d1 = WENO_WEIGHT_1;
        let d2 = WENO_WEIGHT_2;

        // Non-linear weights
        let alpha0 = d0 / (self.epsilon + beta0).powi(2);
        let alpha1 = d1 / (self.epsilon + beta1).powi(2);
        let alpha2 = d2 / (self.epsilon + beta2).powi(2);

        let sum_alpha = alpha0 + alpha1 + alpha2;

        let w0 = alpha0 / sum_alpha;
        let w1 = alpha1 / sum_alpha;
        let w2 = alpha2 / sum_alpha;

        // Final reconstruction
        w0 * q0 + w1 * q1 + w2 * q2
    }
}
