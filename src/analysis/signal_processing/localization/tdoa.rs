//! TDOA (Time-Difference-of-Arrival) Localization
//!
//! Implements source localization from time-delay estimates between sensor pairs.
//!
//! References:
//! - Knapp, C. H., & Carter, G. C. (1976). "The generalized correlation method for estimation of time delay"
//! - Cafforio, C., & Rocca, F. (1976). "Direction determination in seismic signal processing"

use super::config::LocalizationConfig;
use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::signal_processing::localization::{LocalizationProcessor, SourceLocation};

/// TDOA configuration
#[derive(Debug, Clone)]
pub struct TDOAConfig {
    /// Base localization config
    pub config: LocalizationConfig,

    /// Method for time-delay estimation
    pub method: TimeDelayMethod,

    /// Number of Newton-Raphson iterations for refinement
    pub refinement_iterations: usize,

    /// Convergence tolerance for Newton-Raphson
    pub convergence_tolerance: f64,
}

/// Time-delay estimation method
#[derive(Debug, Clone, Copy)]
pub enum TimeDelayMethod {
    /// Cross-correlation at peak
    CrossCorrelation,

    /// Generalized cross-correlation (GCC)
    GeneralizedCrossCorrelation,

    /// Weighted GCC with PHAT weighting
    GCCWithPHAT,
}

impl TDOAConfig {
    /// Create new TDOA configuration
    pub fn new(config: LocalizationConfig, method: TimeDelayMethod) -> Self {
        Self {
            config,
            method,
            refinement_iterations: 5,
            convergence_tolerance: 1e-6,
        }
    }

    /// Set refinement iterations
    pub fn with_refinement_iterations(mut self, iterations: usize) -> Self {
        self.refinement_iterations = iterations;
        self
    }

    /// Set convergence tolerance
    pub fn with_convergence_tolerance(mut self, tolerance: f64) -> Self {
        self.convergence_tolerance = tolerance;
        self
    }
}

impl Default for TDOAConfig {
    fn default() -> Self {
        Self::new(
            LocalizationConfig::default(),
            TimeDelayMethod::CrossCorrelation,
        )
    }
}

/// TDOA processor
#[derive(Debug)]
pub struct TDOAProcessor {
    config: TDOAConfig,
}

impl TDOAProcessor {
    /// Create new TDOA processor
    pub fn new(config: &TDOAConfig) -> KwaversResult<Self> {
        config.config.validate()?;

        if config.refinement_iterations == 0 {
            return Err(KwaversError::InvalidInput(
                "Refinement iterations must be > 0".to_string(),
            ));
        }

        if config.convergence_tolerance <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Convergence tolerance must be > 0".to_string(),
            ));
        }

        Ok(Self {
            config: config.clone(),
        })
    }

    /// Estimate time delays between sensor pairs
    #[allow(dead_code)]
    fn estimate_time_delays(&self, sensor_signals: &[Vec<f64>]) -> Vec<f64> {
        let num_sensors = sensor_signals.len();
        let mut time_delays = vec![0.0; num_sensors * (num_sensors - 1) / 2];

        let mut idx = 0;
        for i in 0..num_sensors {
            for _j in i + 1..num_sensors {
                // Compute cross-correlation between signal i and j
                // For now, placeholder returning zero
                time_delays[idx] = 0.0;
                idx += 1;
            }
        }

        time_delays
    }

    /// Newton-Raphson refinement for source position
    #[allow(dead_code)]
    fn refine_position(
        &self,
        initial_position: &[f64; 3],
        sensor_positions: &[[f64; 3]],
        time_delays: &[f64],
    ) -> KwaversResult<[f64; 3]> {
        let mut position = *initial_position;
        let c = self.config.config.sound_speed;

        for _ in 0..self.config.refinement_iterations {
            #[allow(unused_mut)]
            let mut jacobian = [[0.0; 3]; 16]; // Up to 16 sensors
            let mut residuals = vec![0.0; sensor_positions.len()];

            // Compute Jacobian and residuals
            for (i, sensor_pos) in sensor_positions.iter().enumerate().take(16) {
                let dx = position[0] - sensor_pos[0];
                let dy = position[1] - sensor_pos[1];
                let dz = position[2] - sensor_pos[2];
                let distance = (dx * dx + dy * dy + dz * dz).sqrt();

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

            // Update position (simplified - full implementation needs matrix inversion)
            for i in 0..3 {
                position[i] -= 0.1 * residuals.get(i).copied().unwrap_or(0.0);
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
                "Need at least 3 sensors for 3D localization".to_string(),
            ));
        }

        if time_delays.is_empty() {
            return Err(KwaversError::InvalidInput(
                "No time delay data provided".to_string(),
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

        Ok(SourceLocation {
            position: refined_position,
            confidence: 0.5, // Placeholder
            uncertainty: 0.01,
        })
    }

    fn name(&self) -> &str {
        "TDOA"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tdoa_processor_creation() {
        let config = TDOAConfig::default();
        let result = TDOAProcessor::new(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_tdoa_insufficient_sensors() {
        let processor = TDOAProcessor::new(&TDOAConfig::default()).unwrap();
        let result = processor.localize(&[0.0], &[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]);
        assert!(result.is_err());
    }

    #[test]
    fn test_tdoa_config_builder() {
        let config = TDOAConfig::default()
            .with_refinement_iterations(10)
            .with_convergence_tolerance(1e-8);

        assert_eq!(config.refinement_iterations, 10);
        assert_eq!(config.convergence_tolerance, 1e-8);
    }

    #[test]
    fn test_tdoa_localization() {
        let config = TDOAConfig::default();
        let processor = TDOAProcessor::new(&config).unwrap();

        let sensor_positions = vec![
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
        ];

        let time_delays = vec![0.0, 0.0001, 0.00015, 0.0002];
        let result = processor.localize(&time_delays, &sensor_positions);
        assert!(result.is_ok());
    }
}
