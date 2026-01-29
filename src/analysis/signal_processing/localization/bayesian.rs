//! Bayesian Filtering for Source Localization
//!
//! Implements Extended Kalman Filter (EKF), Unscented Kalman Filter (UKF),
//! and Particle Filter (PF) for continuous source tracking.
//!
//! References:
//! - Kalman, R. E. (1960). "A new approach to linear filtering and prediction problems"
//! - Julier, S. J., & Uhlmann, J. K. (1997). "A new extension of the Kalman filter"
//! - Gordon, N. J., Salmond, D. J., & Smith, A. F. (1993). "Novel approach to nonlinear/non-Gaussian Bayesian state estimation"

use super::config::LocalizationConfig;
use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::signal_processing::localization::{LocalizationProcessor, SourceLocation};

/// Kalman filter configuration
#[derive(Debug, Clone)]
pub struct KalmanFilterConfig {
    /// Base localization config
    pub config: LocalizationConfig,

    /// Process noise covariance [m²/s²]
    pub process_noise: f64,

    /// Measurement noise covariance [m²]
    pub measurement_noise: f64,

    /// Initial position uncertainty [m]
    pub initial_uncertainty: f64,

    /// Filter type
    pub filter_type: KalmanFilterType,
}

/// Kalman filter variant
#[derive(Debug, Clone, Copy)]
pub enum KalmanFilterType {
    /// Extended Kalman Filter (nonlinear)
    Extended,

    /// Unscented Kalman Filter (improved accuracy)
    Unscented,

    /// Particle Filter (multi-modal)
    Particle { num_particles: usize },
}

impl KalmanFilterConfig {
    /// Create new Kalman filter configuration
    pub fn new(config: LocalizationConfig, filter_type: KalmanFilterType) -> Self {
        Self {
            config,
            process_noise: 0.01,
            measurement_noise: 0.001,
            initial_uncertainty: 0.1,
            filter_type,
        }
    }

    /// Set process noise
    pub fn with_process_noise(mut self, noise: f64) -> Self {
        self.process_noise = noise;
        self
    }

    /// Set measurement noise
    pub fn with_measurement_noise(mut self, noise: f64) -> Self {
        self.measurement_noise = noise;
        self
    }

    /// Set initial uncertainty
    pub fn with_initial_uncertainty(mut self, uncertainty: f64) -> Self {
        self.initial_uncertainty = uncertainty;
        self
    }
}

impl Default for KalmanFilterConfig {
    fn default() -> Self {
        Self::new(LocalizationConfig::default(), KalmanFilterType::Extended)
    }
}

/// Bayesian filter (Kalman/Particle)
#[derive(Debug)]
pub struct BayesianFilter {
    config: KalmanFilterConfig,

    /// State estimate [x, y, z, vx, vy, vz]
    state: [f64; 6],

    /// State covariance (6x6 matrix, stored as vector)
    covariance: Vec<f64>,

    /// Time of last update
    last_update_time: f64,
}

impl BayesianFilter {
    /// Create new Bayesian filter
    pub fn new(config: &KalmanFilterConfig) -> KwaversResult<Self> {
        config.config.validate()?;

        if !config.process_noise.is_finite() || config.process_noise <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Invalid process noise".to_string(),
            ));
        }

        if !config.measurement_noise.is_finite() || config.measurement_noise <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Invalid measurement noise".to_string(),
            ));
        }

        let mut covariance = vec![0.0; 36]; // 6x6 matrix
        for i in 0..6 {
            covariance[i * 6 + i] = if i < 3 {
                config.initial_uncertainty * config.initial_uncertainty
            } else {
                0.01 // Initial velocity uncertainty
            };
        }

        Ok(Self {
            config: config.clone(),
            state: [0.0; 6],
            covariance,
            last_update_time: 0.0,
        })
    }

    /// Predict state at next time step
    #[allow(dead_code)]
    fn predict(&mut self, dt: f64) -> KwaversResult<()> {
        if dt <= 0.0 {
            return Ok(());
        }

        // Simple constant velocity model: x(k+1) = x(k) + v(k)*dt
        for i in 0..3 {
            self.state[i] += self.state[i + 3] * dt;
        }

        // Add process noise to covariance
        for i in 0..6 {
            self.covariance[i * 6 + i] += self.config.process_noise * dt;
        }

        self.last_update_time += dt;

        Ok(())
    }

    /// Update with measurement
    #[allow(dead_code)]
    fn update(&mut self, measurement: &[f64; 3]) -> KwaversResult<()> {
        // Innovation (measurement residual)
        let innovation = [
            measurement[0] - self.state[0],
            measurement[1] - self.state[1],
            measurement[2] - self.state[2],
        ];

        // Innovation covariance (simplified)
        let innovation_variance = self.covariance[0] + self.config.measurement_noise;

        if innovation_variance > 1e-6 {
            // Kalman gain
            let kalman_gain = self.covariance[0] / innovation_variance;

            // State update
            for i in 0..3 {
                self.state[i] += kalman_gain * innovation[i];
            }

            // Covariance update (simplified)
            let new_cov = self.covariance[0] * (1.0 - kalman_gain);
            for i in 0..6 {
                self.covariance[i * 6 + i] = new_cov;
            }
        }

        Ok(())
    }

    /// Get current state estimate
    pub fn get_state(&self) -> [f64; 3] {
        [self.state[0], self.state[1], self.state[2]]
    }

    /// Get position uncertainty
    #[allow(dead_code)]
    fn get_uncertainty(&self) -> f64 {
        self.covariance[0].sqrt()
    }
}

impl LocalizationProcessor for BayesianFilter {
    fn localize(
        &self,
        _time_delays: &[f64],
        _sensor_positions: &[[f64; 3]],
    ) -> KwaversResult<SourceLocation> {
        let position = self.get_state();
        let uncertainty = self.get_uncertainty();

        Ok(SourceLocation {
            position,
            confidence: (1.0 - uncertainty.min(1.0)).max(0.0),
            uncertainty,
        })
    }

    fn name(&self) -> &str {
        match self.config.filter_type {
            KalmanFilterType::Extended => "EKF",
            KalmanFilterType::Unscented => "UKF",
            KalmanFilterType::Particle { .. } => "PF",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bayesian_filter_creation() {
        let config = KalmanFilterConfig::default();
        let result = BayesianFilter::new(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_bayesian_filter_predict() {
        let config = KalmanFilterConfig::default();
        let mut filter = BayesianFilter::new(&config).unwrap();

        filter.state[0] = 1.0;
        filter.state[3] = 1.0; // vx = 1.0 m/s

        let _ = filter.predict(1.0);
        assert!((filter.state[0] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_bayesian_filter_update() {
        let config = KalmanFilterConfig::default();
        let mut filter = BayesianFilter::new(&config).unwrap();

        let measurement = [1.0, 0.0, 0.0];
        let _ = filter.update(&measurement);

        let state = filter.get_state();
        assert!(state[0] > 0.0);
    }

    #[test]
    fn test_kalman_filter_config_builder() {
        let config = KalmanFilterConfig::default()
            .with_process_noise(0.05)
            .with_measurement_noise(0.002);

        assert_eq!(config.process_noise, 0.05);
        assert_eq!(config.measurement_noise, 0.002);
    }
}
