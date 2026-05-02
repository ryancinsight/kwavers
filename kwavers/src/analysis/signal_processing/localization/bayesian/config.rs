use super::super::config::LocalizationConfig;

/// Kalman filter configuration
#[derive(Debug, Clone)]
pub struct KalmanFilterConfig {
    pub config: LocalizationConfig,
    /// Process noise spectral density q [m²/s³] (Singer 1970 model)
    pub process_noise: f64,
    /// Measurement noise variance σ_m² [m²]
    pub measurement_noise: f64,
    /// Initial position uncertainty σ₀ [m]
    pub initial_uncertainty: f64,
    pub filter_type: KalmanFilterType,
}

/// Kalman filter variant
#[derive(Debug, Clone, Copy)]
pub enum KalmanFilterType {
    /// Extended Kalman Filter
    Extended,
    /// Unscented Kalman Filter
    Unscented,
    /// Particle Filter
    Particle { num_particles: usize },
}

impl KalmanFilterConfig {
    pub fn new(config: LocalizationConfig, filter_type: KalmanFilterType) -> Self {
        Self {
            config,
            process_noise: 0.01,
            measurement_noise: 0.001,
            initial_uncertainty: 0.1,
            filter_type,
        }
    }

    pub fn with_process_noise(mut self, noise: f64) -> Self {
        self.process_noise = noise;
        self
    }

    pub fn with_measurement_noise(mut self, noise: f64) -> Self {
        self.measurement_noise = noise;
        self
    }

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
