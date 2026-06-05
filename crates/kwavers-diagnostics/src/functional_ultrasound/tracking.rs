//! Real-Time Position Tracking
//!
//! Continuous tracking filter for real-time position updates during navigation.

use kwavers_core::error::KwaversResult;

/// Tracking filter for continuous position estimation
#[derive(Debug)]
pub struct TrackingFilter {
    /// Current position estimate [x, y, z]
    position: [f64; 3],

    /// Position uncertainty (mm)
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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
        self.covariance = (1.0 - gain).mul_add(self.covariance, self.process_noise);

        // Update uncertainty
        self.uncertainty = self.covariance.sqrt();

        Ok(self.position)
    }

    /// Get current position estimate
    #[must_use]
    pub fn get_position(&self) -> [f64; 3] {
        self.position
    }

    /// Get position uncertainty
    #[must_use]
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
        let _filter = TrackingFilter::new().unwrap();
    }

    #[test]
    fn test_tracking_filter_update() {
        let mut filter = TrackingFilter::new().unwrap();
        let measurement = [1.0, 2.0, 3.0];

        filter.update(&measurement).unwrap();

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

    // ─── Kalman filter: exact closed-form verification ────────────────────────

    /// First update computes exact Kalman gain = P₀ / (P₀ + R).
    ///
    /// Initial state: P₀ = 0.01, R = 0.01.
    /// gain = 0.01 / (0.01 + 0.01) = 0.5.
    /// measurement = [2.0, 4.0, 6.0], position₀ = [0, 0, 0].
    /// innovation = measurement.
    /// position₁ = gain · measurement = [1.0, 2.0, 3.0].
    #[test]
    fn tracking_filter_first_update_exact_kalman_position() {
        let mut filter = TrackingFilter::new().unwrap();
        let pos = filter.update(&[2.0, 4.0, 6.0]).unwrap();
        assert!(
            (pos[0] - 1.0).abs() < 1e-12,
            "x: expected 1.0, got {}",
            pos[0]
        );
        assert!(
            (pos[1] - 2.0).abs() < 1e-12,
            "y: expected 2.0, got {}",
            pos[1]
        );
        assert!(
            (pos[2] - 3.0).abs() < 1e-12,
            "z: expected 3.0, got {}",
            pos[2]
        );
    }

    /// Zero measurement leaves position at zero (innovation = 0 → no state change).
    ///
    /// innovation = [0, 0, 0] → gain · innovation = 0 → position stays [0, 0, 0].
    #[test]
    fn tracking_filter_zero_measurement_leaves_position_at_zero() {
        let mut filter = TrackingFilter::new().unwrap();
        let pos = filter.update(&[0.0, 0.0, 0.0]).unwrap();
        assert!(
            pos.iter().all(|&v| v.abs() < 1e-12),
            "zero measurement must yield zero position, got {pos:?}"
        );
    }

    /// After many repeated updates the filter converges to the measurement.
    ///
    /// For any fixed measurement m, the Kalman estimate is a contraction mapping
    /// toward m; repeated application reduces |position − m| monotonically
    /// until the steady-state error is O(Q/R) (process noise / measurement noise).
    /// With Q=0.001, R=0.01, steady-state error < 0.01 after ≥ 50 steps.
    #[test]
    fn tracking_filter_repeated_updates_converge_to_measurement() {
        let mut filter = TrackingFilter::new().unwrap();
        let m = [5.0_f64, -3.0, 1.0];
        for _ in 0..100 {
            filter.update(&m).unwrap();
        }
        let pos = filter.get_position();
        for (i, (&p, &mi)) in pos.iter().zip(m.iter()).enumerate() {
            assert!(
                (p - mi).abs() < 0.05,
                "axis {i}: expected convergence to {mi}, got {p}"
            );
        }
    }

    /// After reset, a fresh update reproduces the same result as the initial update.
    ///
    /// reset() restores P₀ = 0.01; the next update should give gain = 0.5 again.
    #[test]
    fn tracking_filter_reset_restores_reproducible_first_update() {
        let mut filter = TrackingFilter::new().unwrap();
        let pos_initial = filter.update(&[4.0, 0.0, 0.0]).unwrap();

        filter.reset();
        // After reset, re-apply the same measurement.
        let pos_after_reset = filter.update(&[4.0, 0.0, 0.0]).unwrap();

        assert!(
            (pos_initial[0] - pos_after_reset[0]).abs() < 1e-12,
            "reset must reproduce initial-update result; expected {}, got {}",
            pos_initial[0],
            pos_after_reset[0]
        );
        assert_eq!(
            filter.get_position(),
            pos_after_reset,
            "get_position must match last update result"
        );
    }
}
