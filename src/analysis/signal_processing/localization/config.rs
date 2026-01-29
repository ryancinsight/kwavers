//! Localization Configuration

use crate::core::error::KwaversResult;

/// Localization configuration
#[derive(Debug, Clone)]
pub struct LocalizationConfig {
    /// Sensor positions [x, y, z] in meters
    pub sensor_positions: Vec<[f64; 3]>,

    /// Sampling frequency [Hz]
    pub sampling_frequency: f64,

    /// Speed of sound [m/s]
    pub sound_speed: f64,

    /// Time window for analysis [s]
    pub time_window: f64,

    /// Localization search space bounds: (min_x, max_x, min_y, max_y, min_z, max_z) [m]
    pub search_bounds: Option<(f64, f64, f64, f64, f64, f64)>,

    /// Number of spatial grid points in search
    pub grid_resolution: usize,

    /// Confidence threshold (0.0-1.0)
    pub confidence_threshold: f64,
}

impl LocalizationConfig {
    /// Create new localization configuration
    pub fn new(sensor_positions: Vec<[f64; 3]>, sampling_frequency: f64, sound_speed: f64) -> Self {
        Self {
            sensor_positions,
            sampling_frequency,
            sound_speed,
            time_window: 1.0,
            search_bounds: None,
            grid_resolution: 64,
            confidence_threshold: 0.5,
        }
    }

    /// Set time window
    pub fn with_time_window(mut self, window: f64) -> Self {
        self.time_window = window;
        self
    }

    /// Set search bounds
    pub fn with_search_bounds(
        mut self,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        min_z: f64,
        max_z: f64,
    ) -> Self {
        self.search_bounds = Some((min_x, max_x, min_y, max_y, min_z, max_z));
        self
    }

    /// Set grid resolution
    pub fn with_grid_resolution(mut self, resolution: usize) -> Self {
        self.grid_resolution = resolution;
        self
    }

    /// Set confidence threshold
    pub fn with_confidence_threshold(mut self, threshold: f64) -> Self {
        self.confidence_threshold = threshold.max(0.0).min(1.0);
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> KwaversResult<()> {
        if self.sensor_positions.is_empty() {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "No sensor positions specified".to_string(),
            ));
        }

        if !self.sampling_frequency.is_finite() || self.sampling_frequency <= 0.0 {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "Invalid sampling frequency".to_string(),
            ));
        }

        if !self.sound_speed.is_finite() || self.sound_speed <= 0.0 {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "Invalid sound speed".to_string(),
            ));
        }

        if self.time_window <= 0.0 {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "Invalid time window".to_string(),
            ));
        }

        if self.grid_resolution == 0 {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "Grid resolution must be > 0".to_string(),
            ));
        }

        Ok(())
    }

    /// Get wavelength at frequency [m]
    pub fn wavelength(&self, frequency: f64) -> f64 {
        self.sound_speed / frequency
    }

    /// Get time step [s]
    pub fn time_step(&self) -> f64 {
        1.0 / self.sampling_frequency
    }

    /// Get number of samples in time window
    pub fn num_samples(&self) -> usize {
        (self.sampling_frequency * self.time_window) as usize
    }
}

impl Default for LocalizationConfig {
    fn default() -> Self {
        Self::new(
            vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]],
            40.0e6, // 40 MHz
            1540.0, // Tissue sound speed
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = LocalizationConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation() {
        let mut config = LocalizationConfig::default();
        config.sensor_positions.clear();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_builder() {
        let config = LocalizationConfig::default()
            .with_time_window(2.0)
            .with_grid_resolution(128)
            .with_confidence_threshold(0.7);

        assert_eq!(config.time_window, 2.0);
        assert_eq!(config.grid_resolution, 128);
        assert_eq!(config.confidence_threshold, 0.7);
    }

    #[test]
    fn test_wavelength_calculation() {
        let config = LocalizationConfig::default();
        let wavelength = config.wavelength(1.0e6);
        assert!((wavelength - 1.54e-3).abs() < 1e-6);
    }
}
