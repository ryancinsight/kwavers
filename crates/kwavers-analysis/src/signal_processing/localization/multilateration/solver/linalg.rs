//! Linear-algebra helper methods for `Multilateration`.
//!
//! Implements the Levenberg-Marquardt normal-equation solve, JᵀJ assembly,
//! uncertainty estimation via covariance trace, and the underlying 3×3
//! Gaussian-elimination and inversion routines.

use super::Multilateration;
use kwavers_core::error::{KwaversError, KwaversResult};

impl Multilateration {
    pub(super) fn solve_levenberg_marquardt(
        &self,
        jacobian: &[[f64; 3]],
        residuals: &[f64],
        lambda: f64,
    ) -> KwaversResult<[f64; 3]> {
        let mut jtj = self.compute_jtj(jacobian);
        jtj[0][0] += lambda;
        jtj[1][1] += lambda;
        jtj[2][2] += lambda;

        let mut jtr = [0.0; 3];
        for (i, (j_row, &r)) in jacobian.iter().zip(residuals.iter()).enumerate() {
            let weight = if self.config.use_weighted_ls {
                1.0 / (self.sensor_uncertainties[i + 1] * self.sensor_uncertainties[i + 1])
            } else {
                1.0
            };
            for k in 0..3 {
                jtr[k] -= j_row[k] * r * weight;
            }
        }

        self.solve_3x3(&jtj, &jtr)
    }

    pub(super) fn compute_jtj(&self, jacobian: &[[f64; 3]]) -> [[f64; 3]; 3] {
        let mut jtj = [[0.0; 3]; 3];
        for (i, j_row) in jacobian.iter().enumerate() {
            let weight = if self.config.use_weighted_ls {
                1.0 / (self.sensor_uncertainties[i + 1] * self.sensor_uncertainties[i + 1])
            } else {
                1.0
            };
            for k in 0..3 {
                for l in 0..3 {
                    jtj[k][l] += j_row[k] * j_row[l] * weight;
                }
            }
        }
        jtj
    }

    pub(super) fn estimate_uncertainty(&self, position: &[f64; 3]) -> KwaversResult<f64> {
        let jacobian = self.compute_jacobian(position);
        let jtj = self.compute_jtj(&jacobian);
        let cov = self.invert_3x3(&jtj)?;
        let trace = cov[0][0] + cov[1][1] + cov[2][2];
        Ok(trace.sqrt())
    }

    pub(super) fn solve_3x3(&self, a: &[[f64; 3]; 3], b: &[f64; 3]) -> KwaversResult<[f64; 3]> {
        const REGULARIZATION: f64 = 1e-12;

        let mut aug = [[0.0; 4]; 3];
        for i in 0..3 {
            for j in 0..3 {
                aug[i][j] = a[i][j];
            }
            aug[i][i] += REGULARIZATION;
            aug[i][3] = b[i];
        }

        for k in 0..3 {
            let mut max_row = k;
            for i in (k + 1)..3 {
                if aug[i][k].abs() > aug[max_row][k].abs() {
                    max_row = i;
                }
            }
            if max_row != k {
                aug.swap(k, max_row);
            }
            if aug[k][k].abs() < 1e-14 {
                return Err(KwaversError::InvalidInput(
                    "Singular matrix - poor sensor geometry".to_owned(),
                ));
            }
            for i in (k + 1)..3 {
                let factor = aug[i][k] / aug[k][k];
                let row_k = aug[k];
                for (j, value) in aug[i].iter_mut().enumerate().skip(k) {
                    *value -= factor * row_k[j];
                }
            }
        }

        let mut x = [0.0; 3];
        for i in (0..3).rev() {
            let mut sum = aug[i][3];
            for j in (i + 1)..3 {
                sum -= aug[i][j] * x[j];
            }
            x[i] = sum / aug[i][i];
        }

        Ok(x)
    }

    pub(super) fn invert_3x3(&self, a: &[[f64; 3]; 3]) -> KwaversResult<[[f64; 3]; 3]> {
        let det = a[0][2].mul_add(
            a[1][0].mul_add(a[2][1], -(a[1][1] * a[2][0])),
            a[0][0].mul_add(
                a[1][1].mul_add(a[2][2], -(a[1][2] * a[2][1])),
                -(a[0][1] * a[1][0].mul_add(a[2][2], -(a[1][2] * a[2][0]))),
            ),
        );

        if det.abs() < 1e-14 {
            return Err(KwaversError::InvalidInput(
                "Matrix not invertible - degenerate geometry".to_owned(),
            ));
        }

        let inv_det = 1.0 / det;
        let mut inv = [[0.0; 3]; 3];
        inv[0][0] = a[1][1].mul_add(a[2][2], -(a[1][2] * a[2][1])) * inv_det;
        inv[0][1] = a[0][2].mul_add(a[2][1], -(a[0][1] * a[2][2])) * inv_det;
        inv[0][2] = a[0][1].mul_add(a[1][2], -(a[0][2] * a[1][1])) * inv_det;
        inv[1][0] = a[1][2].mul_add(a[2][0], -(a[1][0] * a[2][2])) * inv_det;
        inv[1][1] = a[0][0].mul_add(a[2][2], -(a[0][2] * a[2][0])) * inv_det;
        inv[1][2] = a[0][2].mul_add(a[1][0], -(a[0][0] * a[1][2])) * inv_det;
        inv[2][0] = a[1][0].mul_add(a[2][1], -(a[1][1] * a[2][0])) * inv_det;
        inv[2][1] = a[0][1].mul_add(a[2][0], -(a[0][0] * a[2][1])) * inv_det;
        inv[2][2] = a[0][0].mul_add(a[1][1], -(a[0][1] * a[1][0])) * inv_det;

        Ok(inv)
    }
}
