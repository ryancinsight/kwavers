//! TDOAProcessor implementation.

use super::types::TDOAConfig;
use crate::signal_processing::localization::{LocalizationProcessor, SourceLocation};
use kwavers_core::error::{KwaversError, KwaversResult};

/// TDOA processor
#[derive(Debug)]
pub struct TDOAProcessor {
    pub(super) config: TDOAConfig,
}

impl TDOAProcessor {
    /// Create new TDOA processor
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(config: &TDOAConfig) -> KwaversResult<Self> {
        config.config.validate()?;

        if config.refinement_iterations == 0 {
            return Err(KwaversError::InvalidInput(
                "Refinement iterations must be > 0".to_owned(),
            ));
        }

        if config.convergence_tolerance <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Convergence tolerance must be > 0".to_owned(),
            ));
        }

        Ok(Self {
            config: config.clone(),
        })
    }

    /// Newton-Raphson refinement for source position
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn refine_position(
        &self,
        initial_position: &[f64; 3],
        sensor_positions: &[[f64; 3]],
        time_delays: &[f64],
    ) -> KwaversResult<[f64; 3]> {
        let mut position = *initial_position;
        let c = self.config.config.sound_speed;

        for _ in 0..self.config.refinement_iterations {
            let mut jacobian = [[0.0; 3]; 16]; // Up to 16 sensors
            let mut residuals = vec![0.0; sensor_positions.len()];
            let n_sensors = sensor_positions.len().min(16);

            // Compute Jacobian and residuals
            for (i, sensor_pos) in sensor_positions.iter().enumerate().take(16) {
                let dx = position[0] - sensor_pos[0];
                let dy = position[1] - sensor_pos[1];
                let dz = position[2] - sensor_pos[2];
                let distance = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();

                if distance > 1e-6 {
                    jacobian[i][0] = dx / (distance * c);
                    jacobian[i][1] = dy / (distance * c);
                    jacobian[i][2] = dz / (distance * c);

                    residuals[i] = distance / c - time_delays.get(i).copied().unwrap_or(0.0);
                }
            }

            // Check convergence
            let max_residual = residuals.iter().map(|r| r.abs()).fold(0.0f64, f64::max);
            if max_residual < self.config.convergence_tolerance {
                break;
            }

            // Gauss-Newton update: solve (J^T J) Δx = J^T r for 3×3 system
            let mut jtj = [[0.0f64; 3]; 3];
            let mut jtr = [0.0f64; 3];
            for i in 0..n_sensors {
                for a in 0..3 {
                    jtr[a] += jacobian[i][a] * residuals[i];
                    for b in 0..3 {
                        jtj[a][b] += jacobian[i][a] * jacobian[i][b];
                    }
                }
            }

            // Solve 3×3 system via Cramer's rule
            let det = jtj[0][2].mul_add(
                jtj[1][0].mul_add(jtj[2][1], -(jtj[1][1] * jtj[2][0])),
                jtj[0][0].mul_add(
                    jtj[1][1].mul_add(jtj[2][2], -(jtj[1][2] * jtj[2][1])),
                    -(jtj[0][1] * jtj[1][0].mul_add(jtj[2][2], -(jtj[1][2] * jtj[2][0]))),
                ),
            );

            if det.abs() > 1e-30 {
                let inv_det = 1.0 / det;
                // Cofactor matrix (transposed = inverse * det for symmetric)
                let inv = [
                    [
                        jtj[1][1].mul_add(jtj[2][2], -(jtj[1][2] * jtj[2][1])) * inv_det,
                        jtj[0][2].mul_add(jtj[2][1], -(jtj[0][1] * jtj[2][2])) * inv_det,
                        jtj[0][1].mul_add(jtj[1][2], -(jtj[0][2] * jtj[1][1])) * inv_det,
                    ],
                    [
                        jtj[1][2].mul_add(jtj[2][0], -(jtj[1][0] * jtj[2][2])) * inv_det,
                        jtj[0][0].mul_add(jtj[2][2], -(jtj[0][2] * jtj[2][0])) * inv_det,
                        jtj[0][2].mul_add(jtj[1][0], -(jtj[0][0] * jtj[1][2])) * inv_det,
                    ],
                    [
                        jtj[1][0].mul_add(jtj[2][1], -(jtj[1][1] * jtj[2][0])) * inv_det,
                        jtj[0][1].mul_add(jtj[2][0], -(jtj[0][0] * jtj[2][1])) * inv_det,
                        jtj[0][0].mul_add(jtj[1][1], -(jtj[0][1] * jtj[1][0])) * inv_det,
                    ],
                ];

                // Δx = (J^T J)^-1 J^T r
                for a in 0..3 {
                    let delta: f64 = inv[a].iter().zip(&jtr).map(|(m, r)| m * r).sum();
                    position[a] -= delta;
                }
            }
        }

        Ok(position)
    }
}

impl LocalizationProcessor for TDOAProcessor {
    fn localize(
        &self,
        time_delays: &[f64],
        sensor_positions: &[[f64; 3]],
    ) -> KwaversResult<SourceLocation> {
        if sensor_positions.len() < 3 {
            return Err(KwaversError::InvalidInput(
                "Need at least 3 sensors for 3D localization".to_owned(),
            ));
        }

        if time_delays.is_empty() {
            return Err(KwaversError::InvalidInput(
                "No time delay data provided".to_owned(),
            ));
        }

        // Initial position estimate (centroid of sensors)
        let mut initial_position = [0.0; 3];
        for pos in sensor_positions {
            initial_position[0] += pos[0];
            initial_position[1] += pos[1];
            initial_position[2] += pos[2];
        }
        initial_position[0] /= sensor_positions.len() as f64;
        initial_position[1] /= sensor_positions.len() as f64;
        initial_position[2] /= sensor_positions.len() as f64;

        // Refine with Newton-Raphson
        let refined_position =
            self.refine_position(&initial_position, sensor_positions, time_delays)?;

        // Confidence from post-refinement residual.
        // Re-compute residuals at the refined position and map the RMS residual
        // to a [0, 1] confidence score using an exponential decay.  A residual
        // on the order of one sample period (dt) yields ~37% confidence.
        let c = self.config.config.sound_speed;
        let dt = 1.0 / self.config.config.sampling_frequency;
        let n_sensors = sensor_positions.len().min(16);
        let mut rss = 0.0_f64;
        for (i, sensor_pos) in sensor_positions.iter().enumerate().take(n_sensors) {
            let dx = refined_position[0] - sensor_pos[0];
            let dy = refined_position[1] - sensor_pos[1];
            let dz = refined_position[2] - sensor_pos[2];
            let distance = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();
            let predicted_delay = distance / c;
            let residual = predicted_delay - time_delays.get(i).copied().unwrap_or(0.0);
            rss += residual * residual;
        }
        let rms_residual = (rss / n_sensors.max(1) as f64).sqrt();
        // Map to confidence: exp(-rms_residual / dt) → 1.0 at perfect fit
        let confidence = (-rms_residual / dt.max(1e-30)).exp().clamp(0.0, 1.0);

        Ok(SourceLocation {
            position: refined_position,
            confidence,
            uncertainty: rms_residual * c, // spatial uncertainty in metres
        })
    }

    fn name(&self) -> &str {
        "TDOA"
    }
}
