//! WENO7 (seventh-order) shock limiting implementation.
//!
//! Balsara & Shu (2000). J. Comput. Phys. 160(2):405-452.

use super::types::WENOLimiter;
use kwavers_core::error::KwaversResult;
use leto::Array3;

impl WENOLimiter {
    /// WENO7 limiting — zero-allocation version.
    ///
    /// Reads from immutable `src`, writes limited values into `output`.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn weno7_limit_into(
        &self,
        src: &Array3<f64>,
        shock_indicator: &Array3<f64>,
        output: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let [nx, ny, nz] = src.shape();
        for i in 4..nx - 4 {
            for j in 4..ny - 4 {
                for k in 4..nz - 4 {
                    if shock_indicator[[i, j, k]] > 0.5 {
                        let weno_x = self.weno7_stencil(&[
                            src[[i - 4, j, k]],
                            src[[i - 3, j, k]],
                            src[[i - 2, j, k]],
                            src[[i - 1, j, k]],
                            src[[i, j, k]],
                            src[[i + 1, j, k]],
                            src[[i + 2, j, k]],
                            src[[i + 3, j, k]],
                            src[[i + 4, j, k]],
                        ]);
                        output[[i, j, k]] = weno_x;
                    }
                }
            }
        }
        Ok(())
    }

    /// WENO7 stencil computation.
    /// Based on Balsara & Shu (2000), "Monotonicity Preserving WENO Schemes"
    pub(super) fn weno7_stencil(&self, v: &[f64; 9]) -> f64 {
        // Four candidate stencils for WENO7
        let q0 = -v[0] / 4.0 + 13.0 * v[1] / 12.0 - 23.0 * v[2] / 12.0 + 25.0 * v[3] / 12.0;
        let q1 = v[1] / 12.0 - 5.0 * v[2] / 12.0 + 13.0 * v[3] / 12.0 + v[4] / 4.0;
        let q2 = -v[2] / 12.0 + 7.0 * v[3] / 12.0 + 7.0 * v[4] / 12.0 - v[5] / 12.0;
        let q3 = v[3] / 4.0 + 13.0 * v[4] / 12.0 - 5.0 * v[5] / 12.0 + v[6] / 12.0;

        // Smoothness indicators (more complex for WENO7)
        let beta0 = self.compute_weno7_smoothness(&v[0..5]);
        let beta1 = self.compute_weno7_smoothness(&v[1..6]);
        let beta2 = self.compute_weno7_smoothness(&v[2..7]);
        let beta3 = self.compute_weno7_smoothness(&v[3..8]);

        // Optimal weights for WENO7
        let d0 = 0.05;
        let d1 = 0.45;
        let d2 = 0.45;
        let d3 = 0.05;

        // Non-linear weights
        let alpha0 = d0 / (self.epsilon + beta0).powi(2);
        let alpha1 = d1 / (self.epsilon + beta1).powi(2);
        let alpha2 = d2 / (self.epsilon + beta2).powi(2);
        let alpha3 = d3 / (self.epsilon + beta3).powi(2);

        let sum_alpha = alpha0 + alpha1 + alpha2 + alpha3;

        let w0 = alpha0 / sum_alpha;
        let w1 = alpha1 / sum_alpha;
        let w2 = alpha2 / sum_alpha;
        let w3 = alpha3 / sum_alpha;

        // Final reconstruction
        w0 * q0 + w1 * q1 + w2 * q2 + w3 * q3
    }

    /// Compute WENO7 smoothness indicator for a 5-point stencil.
    /// Based on Jiang & Shu (1996) smoothness indicators.
    pub(super) fn compute_weno7_smoothness(&self, v: &[f64]) -> f64 {
        let d1 = v[1] - v[0];
        let d2 = v[2] - v[1];
        let d3 = v[3] - v[2];
        let d4 = v[4] - v[3];

        let d11 = d2 - d1;
        let d21 = d3 - d2;
        let d31 = d4 - d3;

        let d111 = d21 - d11;
        let d211 = d31 - d21;

        let d1111 = d211 - d111;

        // Sum of squares of derivatives
        (1421461.0 / 2275.0 * d1111).mul_add(
            d1111,
            (781.0 / 20.0 * d111).mul_add(d111, d1.mul_add(d1, 13.0 / 3.0 * d11 * d11)),
        )
    }
}
