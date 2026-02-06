//! Multilateration Source Localization with Advanced TDOA Methods
//!
//! This module implements multilateration algorithms for acoustic source localization
//! using time-difference-of-arrival (TDOA) measurements from sensor arrays. Unlike
//! basic trilateration, multilateration handles overdetermined systems with more
//! sensors than required, improving accuracy through statistical optimization.
//!
//! # Mathematical Foundation
//!
//! ## TDOA Formulation
//!
//! For a source at unknown position **r_s** = [x, y, z] and sensors at known
//! positions **r_i**, the time-of-arrival is:
//!
//! ```text
//! t_i = t_0 + ||r_s - r_i|| / c
//! ```
//!
//! where:
//! - t_0 = unknown emission time
//! - c = sound speed in medium
//! - ||·|| = Euclidean norm
//!
//! Using a reference sensor (typically sensor 1), we eliminate t_0:
//!
//! ```text
//! Δt_i = t_i - t_1 = (||r_s - r_i|| - ||r_s - r_1||) / c
//! ```
//!
//! ## Linearized Least Squares (Foy 1976)
//!
//! Define range differences:
//!
//! ```text
//! R_i = ||r_s - r_i||
//! R_1 = ||r_s - r_1||
//! d_i = c·Δt_i = R_i - R_1  (measured)
//! ```
//!
//! Expanding the range equation:
//!
//! ```text
//! R_i² = (x - x_i)² + (y - y_i)² + (z - z_i)²
//! ```
//!
//! Subtracting R_1² and linearizing around an initial guess **r_0**:
//!
//! ```text
//! R_i² - R_1² = K_i - K_1 - 2(x_i - x_1)x - 2(y_i - y_1)y - 2(z_i - z_1)z
//! ```
//!
//! where K_i = x_i² + y_i² + z_i².
//!
//! Also: R_i² - R_1² = (R_i - R_1)(R_i + R_1) = d_i(d_i + 2R_1)
//!
//! This gives the linear system **A·x = b** where:
//!
//! ```text
//! A_i = [x_i - x_1,  y_i - y_1,  z_i - z_1]
//! b_i = 0.5[d_i(d_i + 2R_1) - K_i + K_1]
//! ```
//!
//! Solution via least squares: **x** = (A^T A)^{-1} A^T b
//!
//! ## Weighted Least Squares
//!
//! For heterogeneous sensor uncertainties σ_i, use weighted least squares:
//!
//! ```text
//! x = (A^T W A)^{-1} A^T W b
//! ```
//!
//! where W = diag(1/σ_1², 1/σ_2², ..., 1/σ_n²)
//!
//! ## Levenberg-Marquardt Refinement
//!
//! The linearization introduces error. Iteratively refine using:
//!
//! ```text
//! (J^T J + λI)·Δx = -J^T·r
//! x_{k+1} = x_k + Δx
//! ```
//!
//! where:
//! - J = Jacobian of residuals
//! - r = residual vector
//! - λ = damping parameter (adaptive)
//!
//! ## Geometric Dilution of Precision (GDOP)
//!
//! GDOP quantifies how sensor geometry amplifies measurement errors:
//!
//! ```text
//! GDOP = √(trace((A^T A)^{-1})) / c
//! ```
//!
//! Lower GDOP indicates better geometry. Optimal when sensors surround source.
//!
//! # References
//!
//! - Foy, W. H. (1976). "Position-Location Solutions by Taylor-Series Estimation"
//!   *IEEE Transactions on Aerospace and Electronic Systems*, AES-12(2), 187-194
//!   DOI: 10.1109/TAES.1976.308294
//!
//! - Smith, J. O., & Abel, J. S. (1987). "Closed-Form Least-Squares Source Location
//!   Estimation from Range-Difference Measurements"
//!   *IEEE Transactions on ASSP*, 35(12), 1661-1669
//!   DOI: 10.1109/TASSP.1987.1165089
//!
//! - Chan, Y. T., & Ho, K. C. (1994). "A Simple and Efficient Estimator for
//!   Hyperbolic Location" *IEEE Trans. Signal Processing*, 42(8), 1905-1915

use crate::analysis::signal_processing::localization::LocalizationResult;
use crate::core::error::{KwaversError, KwaversResult};
use serde::{Deserialize, Serialize};

/// Configuration for multilateration algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultilaterationConfig {
    /// Sound speed in medium (m/s)
    pub sound_speed: f64,

    /// Maximum iterations for Levenberg-Marquardt refinement
    pub max_iterations: usize,

    /// Convergence tolerance for position update (m)
    pub convergence_tolerance: f64,

    /// Initial damping parameter for Levenberg-Marquardt
    pub initial_damping: f64,

    /// Damping adjustment factor (increase on failure, decrease on success)
    pub damping_factor: f64,

    /// Use weighted least squares (requires sensor_uncertainties)
    pub use_weighted_ls: bool,

    /// Initial guess for source position (m), None = centroid
    pub initial_guess: Option<[f64; 3]>,
}

impl Default for MultilaterationConfig {
    fn default() -> Self {
        Self {
            sound_speed: 1540.0,
            max_iterations: 50,
            convergence_tolerance: 1e-6,
            initial_damping: 1e-3,
            damping_factor: 10.0,
            use_weighted_ls: false,
            initial_guess: None,
        }
    }
}

/// Multilateration source localizer for overdetermined TDOA systems
///
/// Solves for source position using least-squares optimization from
/// time-difference-of-arrival measurements. Supports weighted least
/// squares and iterative refinement via Levenberg-Marquardt.
#[derive(Debug)]
pub struct Multilateration {
    config: MultilaterationConfig,
    sensor_positions: Vec<[f64; 3]>,
    sensor_uncertainties: Vec<f64>,
    num_sensors: usize,
}

impl Multilateration {
    /// Create a new multilateration localizer
    ///
    /// # Arguments
    ///
    /// * `sensor_positions` - Sensor positions [[x, y, z], ...] (m)
    /// * `config` - Multilateration configuration
    ///
    /// # Returns
    ///
    /// Configured multilateration localizer
    pub fn new(
        sensor_positions: Vec<[f64; 3]>,
        config: MultilaterationConfig,
    ) -> KwaversResult<Self> {
        let num_sensors = sensor_positions.len();

        if num_sensors < 4 {
            return Err(KwaversError::InvalidInput(
                "Multilateration requires at least 4 sensors for 3D localization".to_string(),
            ));
        }

        if config.sound_speed <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Sound speed must be positive".to_string(),
            ));
        }

        // Default: uniform uncertainty
        let sensor_uncertainties = vec![1.0; num_sensors];

        Ok(Self {
            config,
            sensor_positions,
            sensor_uncertainties,
            num_sensors,
        })
    }

    /// Set sensor measurement uncertainties for weighted least squares
    ///
    /// # Arguments
    ///
    /// * `uncertainties` - Standard deviation of timing error at each sensor (s)
    pub fn set_sensor_uncertainties(&mut self, uncertainties: Vec<f64>) -> KwaversResult<()> {
        if uncertainties.len() != self.num_sensors {
            return Err(KwaversError::InvalidInput(format!(
                "Expected {} uncertainties, got {}",
                self.num_sensors,
                uncertainties.len()
            )));
        }

        for &u in &uncertainties {
            if u <= 0.0 {
                return Err(KwaversError::InvalidInput(
                    "All uncertainties must be positive".to_string(),
                ));
            }
        }

        self.sensor_uncertainties = uncertainties;
        Ok(())
    }

    /// Localize source from time-of-arrival measurements
    ///
    /// Uses linearized least squares with optional Levenberg-Marquardt refinement
    /// for improved accuracy in overdetermined systems.
    ///
    /// # Arguments
    ///
    /// * `arrival_times` - Time of arrival at each sensor (s)
    ///
    /// # Returns
    ///
    /// Source position with uncertainty and quality metrics
    pub fn localize(&self, arrival_times: &[f64]) -> KwaversResult<LocalizationResult> {
        if arrival_times.len() != self.num_sensors {
            return Err(KwaversError::InvalidInput(format!(
                "Expected {} arrival times, got {}",
                self.num_sensors,
                arrival_times.len()
            )));
        }

        // Compute TDOA relative to first sensor
        let ref_time = arrival_times[0];
        let time_diffs: Vec<f64> = arrival_times
            .iter()
            .skip(1)
            .map(|&t| t - ref_time)
            .collect();

        // Convert to range differences
        let range_diffs: Vec<f64> = time_diffs
            .iter()
            .map(|&dt| dt * self.config.sound_speed)
            .collect();

        // Initial guess: centroid or user-provided
        let mut position = self.compute_initial_guess();

        // Levenberg-Marquardt iteration
        let mut lambda = self.config.initial_damping;
        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..self.config.max_iterations {
            iterations = iter + 1;

            // Compute residuals and Jacobian
            let residuals = self.compute_residuals(&position, &range_diffs);
            let jacobian = self.compute_jacobian(&position);

            // Compute residual norm
            let residual_norm = residuals.iter().map(|&r| r * r).sum::<f64>().sqrt();

            if residual_norm < self.config.convergence_tolerance {
                converged = true;
                break;
            }

            // Levenberg-Marquardt step: (J^T W J + λI)δ = -J^T W r
            let update = self.solve_levenberg_marquardt(&jacobian, &residuals, lambda)?;

            // Check if update improves solution
            let new_position = [
                position[0] + update[0],
                position[1] + update[1],
                position[2] + update[2],
            ];

            let new_residuals = self.compute_residuals(&new_position, &range_diffs);
            let new_residual_norm = new_residuals.iter().map(|&r| r * r).sum::<f64>().sqrt();

            if new_residual_norm < residual_norm {
                // Accept update, decrease damping
                position = new_position;
                lambda /= self.config.damping_factor;

                // Check convergence on update magnitude
                let update_norm =
                    (update[0] * update[0] + update[1] * update[1] + update[2] * update[2]).sqrt();
                if update_norm < self.config.convergence_tolerance {
                    converged = true;
                    break;
                }
            } else {
                // Reject update, increase damping
                lambda *= self.config.damping_factor;
            }
        }

        // Final residual and uncertainty
        let final_residuals = self.compute_residuals(&position, &range_diffs);
        let residual = final_residuals.iter().map(|&r| r * r).sum::<f64>().sqrt();

        // Uncertainty from covariance (simplified)
        let uncertainty = self.estimate_uncertainty(&position)?;

        Ok(LocalizationResult {
            position,
            uncertainty,
            residual,
            iterations,
            converged,
        })
    }

    /// Calculate Geometric Dilution of Precision (GDOP)
    ///
    /// GDOP quantifies how sensor geometry affects localization accuracy.
    /// Lower values indicate better geometry (sensors surrounding source).
    ///
    /// # Arguments
    ///
    /// * `source_position` - Estimated source position (m)
    ///
    /// # Returns
    ///
    /// GDOP value (dimensionless, typically 1-10)
    pub fn calculate_gdop(&self, source_position: &[f64; 3]) -> KwaversResult<f64> {
        let jacobian = self.compute_jacobian(source_position);

        // Compute (J^T J)^{-1}
        let jtj = self.compute_jtj(&jacobian);
        let jtj_inv = self.invert_3x3(&jtj)?;

        // GDOP = sqrt(trace(J^T J)^{-1}) / c
        let trace = jtj_inv[0][0] + jtj_inv[1][1] + jtj_inv[2][2];
        let gdop = trace.sqrt() / self.config.sound_speed;

        Ok(gdop)
    }

    /// Compute initial guess for iterative solver
    fn compute_initial_guess(&self) -> [f64; 3] {
        self.config.initial_guess.unwrap_or_else(|| {
            let sum_x: f64 = self.sensor_positions.iter().map(|p| p[0]).sum();
            let sum_y: f64 = self.sensor_positions.iter().map(|p| p[1]).sum();
            let sum_z: f64 = self.sensor_positions.iter().map(|p| p[2]).sum();
            let n = self.num_sensors as f64;
            [sum_x / n, sum_y / n, sum_z / n]
        })
    }

    /// Compute residuals: measured - predicted range differences
    fn compute_residuals(&self, position: &[f64; 3], measured_range_diffs: &[f64]) -> Vec<f64> {
        let ref_pos = &self.sensor_positions[0];
        let ref_range = self.distance(position, ref_pos);

        measured_range_diffs
            .iter()
            .enumerate()
            .map(|(i, &measured)| {
                let sensor_pos = &self.sensor_positions[i + 1];
                let range = self.distance(position, sensor_pos);
                let predicted = range - ref_range;
                measured - predicted
            })
            .collect()
    }

    /// Compute Jacobian matrix: ∂residual/∂position
    fn compute_jacobian(&self, position: &[f64; 3]) -> Vec<[f64; 3]> {
        let ref_pos = &self.sensor_positions[0];
        let ref_range = self.distance(position, ref_pos);

        let ref_dx = (position[0] - ref_pos[0]) / ref_range;
        let ref_dy = (position[1] - ref_pos[1]) / ref_range;
        let ref_dz = (position[2] - ref_pos[2]) / ref_range;

        (1..self.num_sensors)
            .map(|i| {
                let sensor_pos = &self.sensor_positions[i];
                let range = self.distance(position, sensor_pos);

                let dx = (position[0] - sensor_pos[0]) / range;
                let dy = (position[1] - sensor_pos[1]) / range;
                let dz = (position[2] - sensor_pos[2]) / range;

                // ∂(predicted)/∂position = ∂(range_i - range_ref)/∂position
                [-(dx - ref_dx), -(dy - ref_dy), -(dz - ref_dz)]
            })
            .collect()
    }

    /// Solve Levenberg-Marquardt step: (J^T W J + λI)x = -J^T W r
    fn solve_levenberg_marquardt(
        &self,
        jacobian: &[[f64; 3]],
        residuals: &[f64],
        lambda: f64,
    ) -> KwaversResult<[f64; 3]> {
        let mut jtj = self.compute_jtj(jacobian);

        // Add damping: J^T J + λI
        jtj[0][0] += lambda;
        jtj[1][1] += lambda;
        jtj[2][2] += lambda;

        // Compute right-hand side: -J^T r (with optional weighting)
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

        // Solve system
        self.solve_3x3(&jtj, &jtr)
    }

    /// Compute J^T W J (with optional weighting)
    fn compute_jtj(&self, jacobian: &[[f64; 3]]) -> [[f64; 3]; 3] {
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

    /// Estimate position uncertainty from covariance
    fn estimate_uncertainty(&self, position: &[f64; 3]) -> KwaversResult<f64> {
        let jacobian = self.compute_jacobian(position);
        let jtj = self.compute_jtj(&jacobian);
        let cov = self.invert_3x3(&jtj)?;

        // Uncertainty is sqrt of trace of covariance matrix
        let trace = cov[0][0] + cov[1][1] + cov[2][2];
        Ok(trace.sqrt())
    }

    /// Solve 3x3 linear system Ax = b using Gaussian elimination
    fn solve_3x3(&self, a: &[[f64; 3]; 3], b: &[f64; 3]) -> KwaversResult<[f64; 3]> {
        const REGULARIZATION: f64 = 1e-12;

        let mut aug = [[0.0; 4]; 3];
        for i in 0..3 {
            for j in 0..3 {
                aug[i][j] = a[i][j];
            }
            aug[i][i] += REGULARIZATION;
            aug[i][3] = b[i];
        }

        // Forward elimination with partial pivoting
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
                    "Singular matrix - poor sensor geometry".to_string(),
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

        // Back substitution
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

    /// Invert 3x3 matrix (for covariance calculation)
    fn invert_3x3(&self, a: &[[f64; 3]; 3]) -> KwaversResult<[[f64; 3]; 3]> {
        // Compute determinant
        let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

        if det.abs() < 1e-14 {
            return Err(KwaversError::InvalidInput(
                "Matrix not invertible - degenerate geometry".to_string(),
            ));
        }

        let inv_det = 1.0 / det;

        // Compute adjugate and divide by determinant
        let mut inv = [[0.0; 3]; 3];
        inv[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * inv_det;
        inv[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * inv_det;
        inv[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * inv_det;
        inv[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * inv_det;
        inv[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * inv_det;
        inv[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * inv_det;
        inv[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * inv_det;
        inv[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * inv_det;
        inv[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * inv_det;

        Ok(inv)
    }

    /// Euclidean distance between two points
    fn distance(&self, p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
        let dx = p1[0] - p2[0];
        let dy = p1[1] - p2[1];
        let dz = p1[2] - p2[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_multilateration_creation() {
        let sensors = vec![
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.01],
        ];
        let config = MultilaterationConfig::default();
        let multi = Multilateration::new(sensors, config);
        assert!(multi.is_ok());
    }

    #[test]
    fn test_insufficient_sensors() {
        let sensors = vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]];
        let config = MultilaterationConfig::default();
        assert!(Multilateration::new(sensors, config).is_err());
    }

    #[test]
    fn test_set_uncertainties() {
        let sensors = vec![
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.01],
        ];
        let config = MultilaterationConfig::default();
        let mut multi = Multilateration::new(sensors, config).unwrap();

        let uncertainties = vec![1e-9, 1e-9, 1e-9, 1e-9];
        assert!(multi.set_sensor_uncertainties(uncertainties).is_ok());

        // Wrong length
        let bad_uncertainties = vec![1e-9, 1e-9];
        assert!(multi.set_sensor_uncertainties(bad_uncertainties).is_err());
    }

    #[test]
    fn test_localize_symmetric_array() {
        let c = 1500.0;
        // Symmetric array around origin
        let sensors = vec![
            [0.01, 0.0, 0.0],
            [-0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, -0.01, 0.0],
            [0.0, 0.0, 0.01],
        ];

        let config = MultilaterationConfig {
            sound_speed: c,
            initial_guess: Some([0.0, 0.0, 0.0]),
            ..Default::default()
        };
        let multi = Multilateration::new(sensors.clone(), config).unwrap();

        // Source at origin
        let source_pos = [0.0, 0.0, 0.0];
        let arrival_times: Vec<f64> = sensors
            .iter()
            .map(|s| {
                let dx = source_pos[0] - s[0];
                let dy = source_pos[1] - s[1];
                let dz = source_pos[2] - s[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                dist / c
            })
            .collect();

        let result = multi.localize(&arrival_times).unwrap();

        assert!(result.converged);
        assert_relative_eq!(result.position[0], 0.0, epsilon = 1e-4);
        assert_relative_eq!(result.position[1], 0.0, epsilon = 1e-4);
        assert_relative_eq!(result.position[2], 0.0, epsilon = 1e-4);
        assert!(result.residual < 1e-5);
    }

    #[test]
    fn test_localize_off_axis_overdetermined() {
        let c = 1540.0;
        // 6 sensors for overdetermined system
        let sensors = vec![
            [0.0, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.0, 0.02, 0.0],
            [0.0, 0.0, 0.02],
            [0.01, 0.01, 0.0],
            [0.01, 0.0, 0.01],
        ];

        let config = MultilaterationConfig {
            sound_speed: c,
            ..Default::default()
        };
        let multi = Multilateration::new(sensors.clone(), config).unwrap();

        let source_pos = [0.005, 0.005, 0.005];
        let arrival_times: Vec<f64> = sensors
            .iter()
            .map(|s| {
                let dx = source_pos[0] - s[0];
                let dy = source_pos[1] - s[1];
                let dz = source_pos[2] - s[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                dist / c
            })
            .collect();

        let result = multi.localize(&arrival_times).unwrap();

        assert!(result.converged);
        assert_relative_eq!(result.position[0], source_pos[0], epsilon = 1e-4);
        assert_relative_eq!(result.position[1], source_pos[1], epsilon = 1e-4);
        assert_relative_eq!(result.position[2], source_pos[2], epsilon = 1e-4);
    }

    #[test]
    fn test_weighted_least_squares() {
        let c = 1500.0;
        let sensors = vec![
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.01],
            [-0.01, 0.0, 0.0],
        ];

        let config = MultilaterationConfig {
            sound_speed: c,
            use_weighted_ls: true,
            ..Default::default()
        };
        let mut multi = Multilateration::new(sensors.clone(), config).unwrap();

        // Different uncertainties: sensor 1 is more accurate
        let uncertainties = vec![1e-10, 1e-10, 5e-10, 5e-10, 5e-10];
        multi.set_sensor_uncertainties(uncertainties).unwrap();

        let source_pos = [0.002, 0.002, 0.002];
        let arrival_times: Vec<f64> = sensors
            .iter()
            .map(|s| {
                let dx = source_pos[0] - s[0];
                let dy = source_pos[1] - s[1];
                let dz = source_pos[2] - s[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                dist / c
            })
            .collect();

        let result = multi.localize(&arrival_times).unwrap();

        assert!(result.converged);
        assert_relative_eq!(result.position[0], source_pos[0], epsilon = 1e-4);
        assert_relative_eq!(result.position[1], source_pos[1], epsilon = 1e-4);
        assert_relative_eq!(result.position[2], source_pos[2], epsilon = 1e-4);
    }

    #[test]
    fn test_gdop_calculation() {
        let c = 1500.0;
        let sensors = vec![
            [0.01, 0.0, 0.0],
            [-0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, -0.01, 0.0],
            [0.0, 0.0, 0.01],
        ];

        let config = MultilaterationConfig {
            sound_speed: c,
            ..Default::default()
        };
        let multi = Multilateration::new(sensors, config).unwrap();

        let source_pos = [0.0, 0.0, 0.0];
        let gdop = multi.calculate_gdop(&source_pos).unwrap();

        // Symmetric geometry should have reasonable GDOP
        assert!(gdop > 0.0);
        assert!(gdop < 10.0);
    }

    #[test]
    fn test_noisy_measurements() {
        let c = 1540.0;
        let sensors = vec![
            [0.0, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.0, 0.02, 0.0],
            [0.0, 0.0, 0.02],
            [0.01, 0.01, 0.01],
        ];

        let config = MultilaterationConfig {
            sound_speed: c,
            max_iterations: 100,
            ..Default::default()
        };
        let multi = Multilateration::new(sensors.clone(), config).unwrap();

        let source_pos = [0.008, 0.008, 0.008];

        // Add small noise to arrival times
        let noise = [0.0, 1e-9, -1e-9, 0.5e-9, -0.5e-9];
        let arrival_times: Vec<f64> = sensors
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let dx = source_pos[0] - s[0];
                let dy = source_pos[1] - s[1];
                let dz = source_pos[2] - s[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                dist / c + noise[i]
            })
            .collect();

        let result = multi.localize(&arrival_times).unwrap();

        // Should still converge with small noise
        assert!(result.converged);
        assert_relative_eq!(result.position[0], source_pos[0], epsilon = 1e-3);
        assert_relative_eq!(result.position[1], source_pos[1], epsilon = 1e-3);
        assert_relative_eq!(result.position[2], source_pos[2], epsilon = 1e-3);
    }
}
