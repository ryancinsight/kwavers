//! Time-of-Arrival Trilateration for Source Localization
//!
//! This module implements trilateration algorithms that determine source position
//! from time-of-arrival (TOA) measurements at multiple sensors. Trilateration
//! solves the geometric problem of finding the intersection of spheres centered
//! at each sensor.
//!
//! # Mathematical Foundation
//!
//! For a source at unknown position **r_s** and sensors at known positions **r_i**:
//!
//! ```text
//! ||r_s - r_i|| = c·(t_i - t_0)
//! ```
//!
//! where:
//! - c = sound speed
//! - t_i = arrival time at sensor i
//! - t_0 = emission time (unknown)
//!
//! ## Time Difference of Arrival (TDOA)
//!
//! Using relative timing eliminates t_0:
//!
//! ```text
//! ||r_s - r_i|| - ||r_s - r_1|| = c·(t_i - t_1) = c·Δt_i
//! ```
//!
//! This forms a hyperbolic system that can be linearized for solution.
//!
//! # References
//!
//! - Foy, W. H. (1976). "Position-Location Solutions by Taylor-Series Estimation"
//!   *IEEE Trans. Aerospace and Electronic Systems*, 12(2), 187-194
//! - Smith & Abel (1987). "Closed-Form Least-Squares Source Location from Range Differences"
//!   *IEEE Trans. ASSP*, 35(12), 1661-1669

use crate::core::error::{KwaversError, KwaversResult};
use serde::{Deserialize, Serialize};

/// Configuration for trilateration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrilaterationConfig {
    /// Sound speed in medium (m/s)
    pub sound_speed: f64,

    /// Maximum number of iterations for iterative solver
    pub max_iterations: usize,

    /// Convergence tolerance (m)
    pub convergence_tolerance: f64,

    /// Initial guess for source position (m), None = centroid of sensors
    pub initial_guess: Option<[f64; 3]>,
}

impl Default for TrilaterationConfig {
    fn default() -> Self {
        Self {
            sound_speed: 1540.0,
            max_iterations: 100,
            convergence_tolerance: 1e-6,
            initial_guess: None,
        }
    }
}

/// Trilateration source localizer
#[derive(Debug)]
pub struct Trilateration {
    config: TrilaterationConfig,
    sensor_positions: Vec<[f64; 3]>,
    num_sensors: usize,
}

/// Localization result with uncertainty
#[derive(Debug, Clone)]
pub struct LocalizationResult {
    /// Estimated source position (m)
    pub position: [f64; 3],

    /// Position uncertainty (standard deviation, m)
    pub uncertainty: f64,

    /// Residual error (m)
    pub residual: f64,

    /// Number of iterations to converge
    pub iterations: usize,

    /// Whether solution converged
    pub converged: bool,
}

impl Trilateration {
    /// Create a new trilateration localizer
    ///
    /// # Arguments
    ///
    /// * `sensor_positions` - Array positions [[x, y, z], ...]
    /// * `config` - Trilateration configuration
    pub fn new(
        sensor_positions: Vec<[f64; 3]>,
        config: TrilaterationConfig,
    ) -> KwaversResult<Self> {
        let num_sensors = sensor_positions.len();

        if num_sensors < 4 {
            return Err(KwaversError::InvalidInput(
                "Need at least 4 sensors for 3D trilateration".to_string(),
            ));
        }

        if config.sound_speed <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Sound speed must be positive".to_string(),
            ));
        }

        Ok(Self {
            config,
            sensor_positions,
            num_sensors,
        })
    }

    /// Localize source from time-of-arrival measurements
    ///
    /// # Arguments
    ///
    /// * `arrival_times` - Time of arrival at each sensor (s)
    ///
    /// # Returns
    ///
    /// Estimated source position with uncertainty metrics
    pub fn localize(&self, arrival_times: &[f64]) -> KwaversResult<LocalizationResult> {
        if arrival_times.len() != self.num_sensors {
            return Err(KwaversError::InvalidInput(format!(
                "Expected {} arrival times, got {}",
                self.num_sensors,
                arrival_times.len()
            )));
        }

        // Use first sensor as reference for TDOA
        let reference_time = arrival_times[0];
        let time_differences: Vec<f64> = arrival_times
            .iter()
            .skip(1)
            .map(|&t| t - reference_time)
            .collect();

        // Initial guess: centroid of sensors or user-provided
        let initial_pos = self.config.initial_guess.unwrap_or_else(|| {
            let sum_x: f64 = self.sensor_positions.iter().map(|p| p[0]).sum();
            let sum_y: f64 = self.sensor_positions.iter().map(|p| p[1]).sum();
            let sum_z: f64 = self.sensor_positions.iter().map(|p| p[2]).sum();
            let n = self.num_sensors as f64;
            [sum_x / n, sum_y / n, sum_z / n]
        });

        // Gauss-Newton iterative solver
        let mut position = initial_pos;
        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..self.config.max_iterations {
            iterations = iter + 1;

            // Compute residuals and Jacobian
            let (residuals, jacobian) =
                self.compute_residuals_and_jacobian(&position, &time_differences)?;

            // Check convergence
            let residual_norm = residuals.iter().map(|&r| r * r).sum::<f64>().sqrt();

            if residual_norm < self.config.convergence_tolerance {
                converged = true;
                break;
            }

            // Solve normal equations: (J^T J + λI) Δx = -J^T r
            let update = self.solve_least_squares(&jacobian, &residuals)?;

            // Update position
            position[0] += update[0];
            position[1] += update[1];
            position[2] += update[2];

            // Check if update is small
            let update_norm =
                (update[0] * update[0] + update[1] * update[1] + update[2] * update[2]).sqrt();
            if update_norm < self.config.convergence_tolerance {
                converged = true;
                break;
            }
        }

        // Compute final residual and uncertainty
        let (final_residuals, _) =
            self.compute_residuals_and_jacobian(&position, &time_differences)?;
        let residual = final_residuals.iter().map(|&r| r * r).sum::<f64>().sqrt();

        // Simplified uncertainty estimate (should use covariance for rigorous estimate)
        let uncertainty = residual / (self.num_sensors as f64).sqrt();

        Ok(LocalizationResult {
            position,
            uncertainty,
            residual,
            iterations,
            converged,
        })
    }

    /// Compute residuals and Jacobian for Gauss-Newton iteration
    fn compute_residuals_and_jacobian(
        &self,
        position: &[f64; 3],
        time_differences: &[f64],
    ) -> KwaversResult<(Vec<f64>, Vec<[f64; 3]>)> {
        let ref_pos = &self.sensor_positions[0];
        let ref_dist = self.distance(position, ref_pos);

        let mut residuals = Vec::with_capacity(time_differences.len());
        let mut jacobian = Vec::with_capacity(time_differences.len());

        for (i, &td) in time_differences.iter().enumerate() {
            let sensor_pos = &self.sensor_positions[i + 1]; // +1 because we skip reference
            let dist = self.distance(position, sensor_pos);

            // Residual: measured TDOA - predicted TDOA
            let predicted_tdoa = (dist - ref_dist) / self.config.sound_speed;
            let residual = td - predicted_tdoa;
            residuals.push(residual);

            // Jacobian: ∂residual/∂position
            let dx_i = position[0] - sensor_pos[0];
            let dy_i = position[1] - sensor_pos[1];
            let dz_i = position[2] - sensor_pos[2];

            let dx_ref = position[0] - ref_pos[0];
            let dy_ref = position[1] - ref_pos[1];
            let dz_ref = position[2] - ref_pos[2];

            let jac_x = -(dx_i / dist - dx_ref / ref_dist) / self.config.sound_speed;
            let jac_y = -(dy_i / dist - dy_ref / ref_dist) / self.config.sound_speed;
            let jac_z = -(dz_i / dist - dz_ref / ref_dist) / self.config.sound_speed;

            jacobian.push([jac_x, jac_y, jac_z]);
        }

        Ok((residuals, jacobian))
    }

    /// Solve least squares with Levenberg-Marquardt damping: (J^T J + λI)x = J^T b
    fn solve_least_squares(
        &self,
        jacobian: &[[f64; 3]],
        residuals: &[f64],
    ) -> KwaversResult<[f64; 3]> {
        // Compute J^T J (3x3 matrix)
        let mut jtj = [[0.0; 3]; 3];
        for j_row in jacobian {
            for i in 0..3 {
                for j in 0..3 {
                    jtj[i][j] += j_row[i] * j_row[j];
                }
            }
        }

        // Levenberg-Marquardt damping: λ = max(diag(J^T J)) * 1e-6
        // This prevents singularity while keeping updates well-scaled
        let max_diag = jtj[0][0].abs().max(jtj[1][1].abs()).max(jtj[2][2].abs());
        let damping = (max_diag * 1e-6).max(1e-20);
        jtj[0][0] += damping;
        jtj[1][1] += damping;
        jtj[2][2] += damping;

        // Compute -J^T r (3x1 vector) — negative for Gauss-Newton descent direction
        let mut neg_jtr = [0.0; 3];
        for (j_row, &r) in jacobian.iter().zip(residuals.iter()) {
            for i in 0..3 {
                neg_jtr[i] -= j_row[i] * r;
            }
        }

        // Solve 3x3 system: (J^T J + λI) Δx = -J^T r
        let x = self.solve_3x3(&jtj, &neg_jtr)?;

        Ok(x)
    }

    /// Solve 3x3 linear system Ax = b via Gaussian elimination with partial pivoting
    fn solve_3x3(&self, a: &[[f64; 3]; 3], b: &[f64; 3]) -> KwaversResult<[f64; 3]> {
        let mut aug = [[0.0; 4]; 3];
        for i in 0..3 {
            for j in 0..3 {
                aug[i][j] = a[i][j];
            }
            aug[i][3] = b[i];
        }

        // Forward elimination with partial pivoting
        for k in 0..3 {
            // Find pivot
            let mut max_row = k;
            for i in (k + 1)..3 {
                if aug[i][k].abs() > aug[max_row][k].abs() {
                    max_row = i;
                }
            }

            // Swap rows
            if max_row != k {
                aug.swap(k, max_row);
            }

            // Check for singularity
            if aug[k][k].abs() < 1e-30 {
                return Err(KwaversError::InvalidInput(
                    "Singular matrix in trilateration - sensors may be coplanar".to_string(),
                ));
            }

            // Eliminate
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

    /// Compute Euclidean distance between two points
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
    fn test_trilateration_creation() {
        let sensors = vec![
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.01],
        ];
        let config = TrilaterationConfig::default();
        let trilat = Trilateration::new(sensors, config);
        assert!(trilat.is_ok());
    }

    #[test]
    fn test_insufficient_sensors() {
        let sensors = vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]];
        let config = TrilaterationConfig::default();
        assert!(Trilateration::new(sensors, config).is_err());
    }

    #[test]
    fn test_localize_source_at_origin() {
        let c = 1500.0; // m/s
        let sensors = vec![
            [0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 0.01],
            [-0.01, 0.0, 0.0],
        ];

        let config = TrilaterationConfig {
            sound_speed: c,
            initial_guess: Some([0.0, 0.0, 0.0]),
            ..Default::default()
        };
        let trilat = Trilateration::new(sensors.clone(), config).unwrap();

        // Source at origin
        let source_pos = [0.0, 0.0, 0.0];

        // Compute arrival times
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

        let result = trilat.localize(&arrival_times).unwrap();

        // Should converge to origin
        assert!(result.converged);
        assert_relative_eq!(result.position[0], 0.0, epsilon = 1e-4);
        assert_relative_eq!(result.position[1], 0.0, epsilon = 1e-4);
        assert_relative_eq!(result.position[2], 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_localize_off_axis_source() {
        let c = 1540.0;
        let sensors = vec![
            [0.0, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.0, 0.02, 0.0],
            [0.0, 0.0, 0.02],
            [0.01, 0.01, 0.01], // Better geometry - not coplanar
        ];

        let config = TrilaterationConfig {
            sound_speed: c,
            ..Default::default()
        };
        let trilat = Trilateration::new(sensors.clone(), config).unwrap();

        // Source offset from origin (use centroid for better conditioning)
        let source_pos = [0.008, 0.008, 0.008];

        // Compute arrival times
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

        let result = trilat.localize(&arrival_times).unwrap();

        assert!(result.converged);
        assert_relative_eq!(result.position[0], source_pos[0], epsilon = 1e-4);
        assert_relative_eq!(result.position[1], source_pos[1], epsilon = 1e-4);
        assert_relative_eq!(result.position[2], source_pos[2], epsilon = 1e-4);
        assert!(result.residual < 1e-5);
    }
}
