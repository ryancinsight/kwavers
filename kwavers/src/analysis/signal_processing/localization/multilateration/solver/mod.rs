//! `Multilateration` struct and Levenberg-Marquardt TDOA solver.
//!
//! # Mathematical Foundation
//!
//! ## TDOA Formulation (Foy 1976)
//!
//! For a source at **r_s** = [x, y, z] and sensors at **r_i**, TDOA relative to
//! sensor 1 gives range differences d_i = c·Δt_i = R_i − R_1. Linearized system:
//!
//! ```text
//! A_i = [x_i − x_1,  y_i − y_1,  z_i − z_1]
//! b_i = 0.5[d_i(d_i + 2R_1) − K_i + K_1]
//! ```
//!
//! Solution via weighted least squares (J^T W J + λI)δ = −J^T W r.
//!
//! ## GDOP
//!
//! ```text
//! GDOP = √(trace((A^T A)^{-1})) / c
//! ```

mod geometry;
mod linalg;

use super::types::MultilaterationConfig;
use crate::analysis::signal_processing::localization::LocalizationResult;
use crate::core::error::{KwaversError, KwaversResult};

/// Multilateration source localizer for overdetermined TDOA systems
///
/// Solves for source position using least-squares optimization from
/// time-difference-of-arrival measurements. Supports weighted least
/// squares and iterative refinement via Levenberg-Marquardt.
#[derive(Debug)]
pub struct Multilateration {
    pub(super) config: MultilaterationConfig,
    pub(super) sensor_positions: Vec<[f64; 3]>,
    pub(super) sensor_uncertainties: Vec<f64>,
    pub(super) num_sensors: usize,
}

impl Multilateration {
    /// Create a new multilateration localizer
    ///
    /// # Arguments
    ///
    /// * `sensor_positions` - Sensor positions [[x, y, z], ...] (m)
    /// * `config` - Multilateration configuration
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid
    ///   or out-of-range input parameters is violated.
    ///
    pub fn new(
        sensor_positions: Vec<[f64; 3]>,
        config: MultilaterationConfig,
    ) -> KwaversResult<Self> {
        let num_sensors = sensor_positions.len();

        if num_sensors < 4 {
            return Err(KwaversError::InvalidInput(
                "Multilateration requires at least 4 sensors for 3D localization".to_owned(),
            ));
        }

        if config.sound_speed <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Sound speed must be positive".to_owned(),
            ));
        }

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
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid
    ///   or out-of-range input parameters is violated.
    ///
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
                    "All uncertainties must be positive".to_owned(),
                ));
            }
        }

        self.sensor_uncertainties = uncertainties;
        Ok(())
    }

    /// Localize source from time-of-arrival measurements
    ///
    /// Uses linearized least squares with Levenberg-Marquardt refinement.
    ///
    /// # Arguments
    ///
    /// * `arrival_times` - Time of arrival at each sensor (s)
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid
    ///   or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn localize(&self, arrival_times: &[f64]) -> KwaversResult<LocalizationResult> {
        if arrival_times.len() != self.num_sensors {
            return Err(KwaversError::InvalidInput(format!(
                "Expected {} arrival times, got {}",
                self.num_sensors,
                arrival_times.len()
            )));
        }

        let ref_time = arrival_times[0];
        let range_diffs: Vec<f64> = arrival_times
            .iter()
            .skip(1)
            .map(|&t| (t - ref_time) * self.config.sound_speed)
            .collect();

        let mut position = self.compute_initial_guess();
        let mut lambda = self.config.initial_damping;
        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..self.config.max_iterations {
            iterations = iter + 1;

            let residuals = self.compute_residuals(&position, &range_diffs);
            let jacobian = self.compute_jacobian(&position);

            let residual_norm = residuals.iter().map(|&r| r * r).sum::<f64>().sqrt();

            if residual_norm < self.config.convergence_tolerance {
                converged = true;
                break;
            }

            let update = self.solve_levenberg_marquardt(&jacobian, &residuals, lambda)?;

            let new_position = [
                position[0] + update[0],
                position[1] + update[1],
                position[2] + update[2],
            ];

            let new_residuals = self.compute_residuals(&new_position, &range_diffs);
            let new_residual_norm = new_residuals.iter().map(|&r| r * r).sum::<f64>().sqrt();

            if new_residual_norm < residual_norm {
                position = new_position;
                lambda /= self.config.damping_factor;

                let update_norm = update[2]
                    .mul_add(
                        update[2],
                        update[0].mul_add(update[0], update[1] * update[1]),
                    )
                    .sqrt();
                if update_norm < self.config.convergence_tolerance {
                    converged = true;
                    break;
                }
            } else {
                lambda *= self.config.damping_factor;
            }
        }

        let final_residuals = self.compute_residuals(&position, &range_diffs);
        let residual = final_residuals.iter().map(|&r| r * r).sum::<f64>().sqrt();
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
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn calculate_gdop(&self, source_position: &[f64; 3]) -> KwaversResult<f64> {
        let jacobian = self.compute_jacobian(source_position);
        let jtj = self.compute_jtj(&jacobian);
        let jtj_inv = self.invert_3x3(&jtj)?;
        let trace = jtj_inv[0][0] + jtj_inv[1][1] + jtj_inv[2][2];
        Ok(trace.sqrt() / self.config.sound_speed)
    }

    fn compute_initial_guess(&self) -> [f64; 3] {
        self.config.initial_guess.unwrap_or_else(|| {
            let sum_x: f64 = self.sensor_positions.iter().map(|p| p[0]).sum();
            let sum_y: f64 = self.sensor_positions.iter().map(|p| p[1]).sum();
            let sum_z: f64 = self.sensor_positions.iter().map(|p| p[2]).sum();
            let n = self.num_sensors as f64;
            [sum_x / n, sum_y / n, sum_z / n]
        })
    }
}
