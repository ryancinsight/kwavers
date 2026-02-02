//! Real-Time Position Tracking
//!
//! Continuous tracking filter for real-time position updates during navigation.

use crate::core::error::KwaversResult;

/// Tracking filter for continuous position estimation
#[derive(Debug)]
pub struct TrackingFilter {
    /// Current position estimate [x, y, z]
    position: [f64; 3],

    /// Position uncertainty [mm]
    uncertainty: f64,

    /// Filter state covariance
    covariance: f64,

    /// Process noise
    process_noise: f64,

    /// Measurement noise
    measurement_noise: f64,
}

impl TrackingFilter {
    /// Create new tracking filter
    pub fn new() -> KwaversResult<Self> {
        Ok(Self {
            position: [0.0, 0.0, 0.0],
            uncertainty: 0.1,
            covariance: 0.01,
            process_noise: 0.001,
            measurement_noise: 0.01,
        })
    }

    /// Update filter with measurement
    pub fn update(&mut self, measurement: &[f64; 3]) -> KwaversResult<[f64; 3]> {
        // Simple Kalman filter update
        let innovation = [
            measurement[0] - self.position[0],
            measurement[1] - self.position[1],
            measurement[2] - self.position[2],
        ];

        let innovation_variance = self.covariance + self.measurement_noise;

        // Kalman gain
        let gain = if innovation_variance > 1e-6 {
            self.covariance / innovation_variance
        } else {
            0.0
        };

        // State update
        for (i, item) in innovation.iter().enumerate() {
            self.position[i] += gain * item;
        }

        // Covariance update
        self.covariance = (1.0 - gain) * self.covariance + self.process_noise;

        // Update uncertainty
        self.uncertainty = self.covariance.sqrt();

        Ok(self.position)
    }

    /// Get current position estimate
    pub fn get_position(&self) -> [f64; 3] {
        self.position
    }

    /// Get position uncertainty
    pub fn get_uncertainty(&self) -> f64 {
        self.uncertainty
    }

    /// Reset filter to initial state
    pub fn reset(&mut self) {
        self.position = [0.0, 0.0, 0.0];
        self.uncertainty = 0.1;
        self.covariance = 0.01;
    }

    /// Set process noise (higher = more dynamic)
    pub fn set_process_noise(&mut self, noise: f64) {
        self.process_noise = noise.max(0.0);
    }

    /// Set measurement noise (higher = less trust measurements)
    pub fn set_measurement_noise(&mut self, noise: f64) {
        self.measurement_noise = noise.max(0.0);
    }
}

impl Default for TrackingFilter {
    fn default() -> Self {
        Self::new().expect("Failed to create default TrackingFilter")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracking_filter_creation() {
        let result = TrackingFilter::new();
        assert!(result.is_ok());
    }

    #[test]
    fn test_tracking_filter_update() {
        let mut filter = TrackingFilter::new().unwrap();
        let measurement = [1.0, 2.0, 3.0];

        let result = filter.update(&measurement);
        assert!(result.is_ok());

        let position = filter.get_position();
        assert!(position[0] > 0.0);
    }

    #[test]
    fn test_tracking_filter_uncertainty() {
        let mut filter = TrackingFilter::new().unwrap();
        let initial_uncertainty = filter.get_uncertainty();

        let measurement = [1.0, 1.0, 1.0];
        let _ = filter.update(&measurement);

        let updated_uncertainty = filter.get_uncertainty();
        assert!(updated_uncertainty <= initial_uncertainty);
    }

    #[test]
    fn test_tracking_filter_reset() {
        let mut filter = TrackingFilter::new().unwrap();

        let measurement = [1.0, 2.0, 3.0];
        let _ = filter.update(&measurement);

        assert!((filter.get_position()[0] - 0.0).abs() > 0.001);

        filter.reset();
        assert_eq!(filter.get_position(), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_tracking_filter_noise_settings() {
        let mut filter = TrackingFilter::new().unwrap();

        filter.set_process_noise(0.05);
        filter.set_measurement_noise(0.02);

        let measurement = [1.0, 1.0, 1.0];
        let _ = filter.update(&measurement);

        assert!(filter.get_position()[0] > 0.0);
    }
}
