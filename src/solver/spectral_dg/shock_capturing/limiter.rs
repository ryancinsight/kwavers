//! WENO-based shock limiting for spectral DG methods
//!
//! Implements Weighted Essentially Non-Oscillatory (WENO) limiters for handling shocks.

use crate::error::{ConfigError, KwaversError, KwaversResult, ValidationError};
use ndarray::Array3;

use crate::physics::constants::numerical::{
    SHOCK_DETECTION_THRESHOLD, STENCIL_COEFF_1_4, WENO_EPSILON, WENO_WEIGHT_0, WENO_WEIGHT_1,
    WENO_WEIGHT_2,
};

/// WENO-based shock limiter
#[derive(Debug, Clone)]
pub struct WENOLimiter {
    /// WENO order (3, 5, or 7)
    order: usize,
    /// Small parameter to avoid division by zero
    epsilon: f64,
    /// Power parameter for smoothness indicators
    p: f64,
    /// Threshold for shock detection
    shock_threshold: f64,
}

impl WENOLimiter {
    pub fn new(order: usize) -> KwaversResult<Self> {
        if order != 3 && order != 5 && order != 7 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "weno_order".to_string(),
                value: order.to_string(),
                constraint: "WENO order must be 3, 5, or 7".to_string(),
            }));
        }

        Ok(Self {
            order,
            epsilon: WENO_EPSILON,
            p: 2.0,
            shock_threshold: SHOCK_DETECTION_THRESHOLD,
        })
    }

    /// Apply WENO limiting to a field
    pub fn limit_field(
        &self,
        field: &Array3<f64>,
        shock_indicator: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let mut limited = field.clone();

        match self.order {
            3 => self.weno3_limit(&mut limited, shock_indicator)?,
            5 => self.weno5_limit(&mut limited, shock_indicator)?,
            7 => self.weno7_limit(&mut limited, shock_indicator)?,
            _ => {
                return Err(KwaversError::Validation(ValidationError::FieldValidation {
                    field: "weno_order".to_string(),
                    value: self.order.to_string(),
                    constraint: "must be 3, 5, or 7".to_string(),
                }));
            }
        }

        Ok(limited)
    }

    /// WENO3 limiting (third-order WENO)
    fn weno3_limit(
        &self,
        field: &mut Array3<f64>,
        shock_indicator: &Array3<f64>,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = field.dim();
        let mut limited_field = field.clone();

        for i in 2..nx - 2 {
            for j in 2..ny - 2 {
                for k in 2..nz - 2 {
                    if shock_indicator[[i, j, k]] > 0.5 {
                        // Apply WENO3 in each direction
                        let weno_x = self.weno3_stencil(&[
                            field[[i - 2, j, k]],
                            field[[i - 1, j, k]],
                            field[[i, j, k]],
                            field[[i + 1, j, k]],
                            field[[i + 2, j, k]],
                        ]);
                        let weno_y = self.weno3_stencil(&[
                            field[[i, j - 2, k]],
                            field[[i, j - 1, k]],
                            field[[i, j, k]],
                            field[[i, j + 1, k]],
                            field[[i, j + 2, k]],
                        ]);
                        let weno_z = self.weno3_stencil(&[
                            field[[i, j, k - 2]],
                            field[[i, j, k - 1]],
                            field[[i, j, k]],
                            field[[i, j, k + 1]],
                            field[[i, j, k + 2]],
                        ]);

                        // Average the limited values
                        limited_field[[i, j, k]] = (weno_x + weno_y + weno_z) / 3.0;
                    }
                }
            }
        }

        field.assign(&limited_field);
        Ok(())
    }

    /// WENO3 stencil computation
    fn weno3_stencil(&self, v: &[f64; 5]) -> f64 {
        // Three candidate stencils
        let q0 = v[0] / 3.0 - 7.0 * v[1] / 6.0 + 11.0 * v[2] / 6.0;
        let q1 = -v[1] / 6.0 + 5.0 * v[2] / 6.0 + v[3] / 3.0;
        let q2 = v[2] / 3.0 + 5.0 * v[3] / 6.0 - v[4] / 6.0;

        // Smoothness indicators (Jiang-Shu)
        let beta0 = 13.0 / 12.0 * (v[0] - 2.0 * v[1] + v[2]).powi(2)
            + STENCIL_COEFF_1_4 * (v[0] - 4.0 * v[1] + 3.0 * v[2]).powi(2);
        let beta1 = 13.0 / 12.0 * (v[1] - 2.0 * v[2] + v[3]).powi(2)
            + STENCIL_COEFF_1_4 * (v[1] - v[3]).powi(2);
        let beta2 = 13.0 / 12.0 * (v[2] - 2.0 * v[3] + v[4]).powi(2)
            + STENCIL_COEFF_1_4 * (3.0 * v[2] - 4.0 * v[3] + v[4]).powi(2);

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

    /// WENO5 limiting implementation
    fn weno5_limit(
        &self,
        field: &mut Array3<f64>,
        shock_indicator: &Array3<f64>,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = field.dim();

        // Process each direction
        for direction in 0..3 {
            match direction {
                0 => self.weno5_limit_x(field, shock_indicator, nx, ny, nz)?,
                1 => self.weno5_limit_y(field, shock_indicator, nx, ny, nz)?,
                2 => self.weno5_limit_z(field, shock_indicator, nx, ny, nz)?,
                _ => continue,
            }
        }

        Ok(())
    }

    /// WENO5 limiting in x-direction
    fn weno5_limit_x(
        &self,
        field: &mut Array3<f64>,
        shock_indicator: &Array3<f64>,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> KwaversResult<()> {
        // Collect indices where shock indicator exceeds threshold
        let indices: Vec<(usize, usize, usize)> = (0..nx)
            .flat_map(|i| (0..ny).flat_map(move |j| (0..nz).map(move |k| (i, j, k))))
            .filter(|&(i, j, k)| {
                i >= 2 && i < nx - 2 && shock_indicator[[i, j, k]] > self.shock_threshold
            })
            .collect();

        // Process each index
        for (i, j, k) in indices {
            // Extract stencil values
            let v = [
                field[[i.saturating_sub(2), j, k]],
                field[[i.saturating_sub(1), j, k]],
                field[[i, j, k]],
                field[[i.min(nx - 1).saturating_add(1), j, k]],
                field[[(i + 2).min(nx - 1), j, k]],
            ];

            field[[i, j, k]] = self.compute_weno5_value(&v);
        }

        Ok(())
    }

    /// WENO5 limiting in y-direction
    fn weno5_limit_y(
        &self,
        field: &mut Array3<f64>,
        shock_indicator: &Array3<f64>,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> KwaversResult<()> {
        // Collect indices where shock indicator exceeds threshold
        let indices: Vec<(usize, usize, usize)> = (0..nx)
            .flat_map(|i| (0..ny).flat_map(move |j| (0..nz).map(move |k| (i, j, k))))
            .filter(|&(i, j, k)| {
                j >= 2 && j < ny - 2 && shock_indicator[[i, j, k]] > self.shock_threshold
            })
            .collect();

        // Process each index
        for (i, j, k) in indices {
            // Extract stencil values
            let v = [
                field[[i, j.saturating_sub(2), k]],
                field[[i, j.saturating_sub(1), k]],
                field[[i, j, k]],
                field[[i, j.min(ny - 1).saturating_add(1), k]],
                field[[i, (j + 2).min(ny - 1), k]],
            ];

            field[[i, j, k]] = self.compute_weno5_value(&v);
        }

        Ok(())
    }

    /// WENO5 limiting in z-direction
    fn weno5_limit_z(
        &self,
        field: &mut Array3<f64>,
        shock_indicator: &Array3<f64>,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> KwaversResult<()> {
        // Collect indices where shock indicator exceeds threshold
        let indices: Vec<(usize, usize, usize)> = (0..nx)
            .flat_map(|i| (0..ny).flat_map(move |j| (0..nz).map(move |k| (i, j, k))))
            .filter(|&(i, j, k)| {
                k >= 2 && k < nz - 2 && shock_indicator[[i, j, k]] > self.shock_threshold
            })
            .collect();

        // Process each index
        for (i, j, k) in indices {
            // Extract stencil values
            let v = [
                field[[i, j, k.saturating_sub(2)]],
                field[[i, j, k.saturating_sub(1)]],
                field[[i, j, k]],
                field[[i, j, k.min(nz - 1).saturating_add(1)]],
                field[[i, j, (k + 2).min(nz - 1)]],
            ];

            field[[i, j, k]] = self.compute_weno5_value(&v);
        }

        Ok(())
    }

    /// Compute WENO5 reconstruction value from stencil
    fn compute_weno5_value(&self, v: &[f64; 5]) -> f64 {
        // Three stencils for WENO5
        let p0 = (2.0 * v[0] - 7.0 * v[1] + 11.0 * v[2]) / 6.0;
        let p1 = (-v[1] + 5.0 * v[2] + 2.0 * v[3]) / 6.0;
        let p2 = (2.0 * v[2] + 5.0 * v[3] - v[4]) / 6.0;

        // Smoothness indicators
        let beta0 = 13.0 / 12.0 * (v[0] - 2.0 * v[1] + v[2]).powi(2)
            + 0.25 * (v[0] - 4.0 * v[1] + 3.0 * v[2]).powi(2);
        let beta1 = 13.0 / 12.0 * (v[1] - 2.0 * v[2] + v[3]).powi(2) + 0.25 * (v[1] - v[3]).powi(2);
        let beta2 = 13.0 / 12.0 * (v[2] - 2.0 * v[3] + v[4]).powi(2)
            + 0.25 * (3.0 * v[2] - 4.0 * v[3] + v[4]).powi(2);

        // Optimal weights
        let d0 = 0.1;
        let d1 = 0.6;
        let d2 = 0.3;

        // WENO weights
        let alpha0 = d0 / (self.epsilon + beta0).powi(2);
        let alpha1 = d1 / (self.epsilon + beta1).powi(2);
        let alpha2 = d2 / (self.epsilon + beta2).powi(2);
        let sum_alpha = alpha0 + alpha1 + alpha2;

        let w0 = alpha0 / sum_alpha;
        let w1 = alpha1 / sum_alpha;
        let w2 = alpha2 / sum_alpha;

        // WENO5 reconstruction
        w0 * p0 + w1 * p1 + w2 * p2
    }

    /// WENO7 limiting (seventh-order WENO)
    fn weno7_limit(
        &self,
        field: &mut Array3<f64>,
        shock_indicator: &Array3<f64>,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = field.dim();
        let mut limited_field = field.clone();

        // WENO7 requires wider stencil (9 points)
        for i in 4..nx - 4 {
            for j in 4..ny - 4 {
                for k in 4..nz - 4 {
                    if shock_indicator[[i, j, k]] > 0.5 {
                        // Apply WENO7 in each direction
                        let weno_x = self.weno7_stencil(&[
                            field[[i - 4, j, k]],
                            field[[i - 3, j, k]],
                            field[[i - 2, j, k]],
                            field[[i - 1, j, k]],
                            field[[i, j, k]],
                            field[[i + 1, j, k]],
                            field[[i + 2, j, k]],
                            field[[i + 3, j, k]],
                            field[[i + 4, j, k]],
                        ]);

                        limited_field[[i, j, k]] = weno_x;
                    }
                }
            }
        }

        field.assign(&limited_field);
        Ok(())
    }

    /// WENO7 stencil computation
    /// Based on Balsara & Shu (2000), "Monotonicity Preserving WENO Schemes"
    fn weno7_stencil(&self, v: &[f64; 9]) -> f64 {
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

    /// Compute WENO7 smoothness indicator for a 5-point stencil
    fn compute_weno7_smoothness(&self, v: &[f64]) -> f64 {
        // Based on Jiang & Shu (1996) smoothness indicators
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
        d1 * d1
            + 13.0 / 3.0 * d11 * d11
            + 781.0 / 20.0 * d111 * d111
            + 1421461.0 / 2275.0 * d1111 * d1111
    }
}
