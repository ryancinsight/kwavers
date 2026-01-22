//! Autocorrelation-Based Velocity Estimation
//!
//! Implements the Kasai autocorrelation method for real-time color flow imaging.
//! This is the industry-standard approach for 2D Doppler velocity estimation.
//!
//! # Algorithm
//!
//! Given an ensemble of `N` complex I/Q signals at a spatial location:
//!
//! 1. Compute lag-1 autocorrelation: `R₁ = Σ(I_n * conj(I_{n+1}))`
//! 2. Extract phase: `φ = arctan(Im(R₁) / Re(R₁))`
//! 3. Convert to velocity: `v = (φ * c) / (4π * f₀ * T_prf * cos(θ))`
//!
//! # References
//!
//! - Kasai, C. et al. (1985). "Real-time two-dimensional blood flow imaging using an autocorrelation technique".
//!   *IEEE Transactions on Sonics and Ultrasonics*, 32(3), 458-464.
//! - Loupas, T. et al. (1995). "An axial velocity estimator for ultrasound blood flow imaging".
//!   *IEEE Transactions on Ultrasonics, Ferroelectrics and Frequency Control*, 42(4), 672-688.

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{Array2, ArrayView3};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Configuration for autocorrelation velocity estimation
#[derive(Debug, Clone)]
pub struct AutocorrelationConfig {
    /// Transmitted center frequency (Hz)
    pub center_frequency: f64,

    /// Pulse repetition frequency (Hz)
    pub prf: f64,

    /// Speed of sound in tissue (m/s)
    pub speed_of_sound: f64,

    /// Beam-to-flow angle (radians)
    /// 0 = flow parallel to beam, π/2 = perpendicular
    pub beam_angle: f64,

    /// Ensemble size (number of pulses to average)
    pub ensemble_size: usize,

    /// Variance threshold for flow detection
    /// Velocities with variance above this are rejected
    pub variance_threshold: f64,
}

impl Default for AutocorrelationConfig {
    fn default() -> Self {
        Self {
            center_frequency: 5.0e6, // 5 MHz
            prf: 4e3,                // 4 kHz
            speed_of_sound: 1540.0,  // m/s
            beam_angle: 0.0,         // Parallel to beam
            ensemble_size: 10,
            variance_threshold: 0.5, // Reject high-variance estimates
        }
    }
}

impl AutocorrelationConfig {
    /// Create configuration for cardiac imaging
    pub fn cardiac() -> Self {
        Self {
            center_frequency: 2.5e6,
            prf: 3e3,
            ensemble_size: 12,
            ..Default::default()
        }
    }

    /// Create configuration for vascular imaging
    pub fn vascular() -> Self {
        Self {
            center_frequency: 7.5e6,
            prf: 5e3,
            ensemble_size: 8,
            ..Default::default()
        }
    }

    /// Calculate Nyquist velocity limit (maximum unambiguous velocity)
    pub fn nyquist_velocity(&self) -> f64 {
        (self.prf * self.speed_of_sound) / (4.0 * self.center_frequency * self.beam_angle.cos())
    }

    /// Calculate velocity resolution (minimum detectable velocity difference)
    pub fn velocity_resolution(&self) -> f64 {
        self.speed_of_sound / (2.0 * self.center_frequency * self.prf * (self.ensemble_size as f64))
    }
}

/// Autocorrelation-based velocity estimator
///
/// Estimates blood flow velocities from complex I/Q ultrasound signals using
/// the Kasai autocorrelation method.
#[derive(Debug, Clone)]
pub struct AutocorrelationEstimator {
    config: AutocorrelationConfig,
}

impl AutocorrelationEstimator {
    /// Create a new autocorrelation estimator
    pub fn new(config: AutocorrelationConfig) -> Self {
        Self { config }
    }

    /// Estimate velocity from complex I/Q signal ensemble
    ///
    /// # Arguments
    ///
    /// * `iq_data` - Complex I/Q signals, shape: (ensemble_size, n_depths, n_beams)
    ///
    /// # Returns
    ///
    /// Tuple of (velocity, variance) arrays, each with shape (n_depths, n_beams)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use kwavers::clinical::imaging::doppler::{AutocorrelationEstimator, AutocorrelationConfig};
    /// use ndarray::Array3;
    /// use num_complex::Complex64;
    ///
    /// let config = AutocorrelationConfig::vascular();
    /// let estimator = AutocorrelationEstimator::new(config);
    ///
    /// // I/Q data: 10 pulses × 256 depths × 64 beams
    /// let iq_data = Array3::<Complex64>::zeros((10, 256, 64));
    ///
    /// let (velocity, variance) = estimator.estimate(&iq_data.view())?;
    /// ```
    pub fn estimate(
        &self,
        iq_data: &ArrayView3<Complex64>,
    ) -> KwaversResult<(Array2<f64>, Array2<f64>)> {
        let (ensemble_size, n_depths, n_beams) = iq_data.dim();

        if ensemble_size < 2 {
            return Err(KwaversError::Validation(
                crate::core::error::ValidationError::InvalidParameter {
                    parameter: "ensemble_size".to_string(),
                    reason: "Ensemble size must be at least 2 for autocorrelation".to_string(),
                },
            ));
        }

        if ensemble_size != self.config.ensemble_size {
            return Err(KwaversError::Validation(
                crate::core::error::ValidationError::InvalidParameter {
                    parameter: "ensemble_size".to_string(),
                    reason: format!(
                        "Expected ensemble size {}, got {}",
                        self.config.ensemble_size, ensemble_size
                    ),
                },
            ));
        }

        let mut velocity = Array2::<f64>::zeros((n_depths, n_beams));
        let mut variance = Array2::<f64>::zeros((n_depths, n_beams));

        // Process each spatial location
        for depth in 0..n_depths {
            for beam in 0..n_beams {
                let (v, var) = self.estimate_at_point(iq_data, depth, beam);
                velocity[[depth, beam]] = v;
                variance[[depth, beam]] = var;
            }
        }

        Ok((velocity, variance))
    }

    /// Estimate velocity at a single spatial point
    fn estimate_at_point(
        &self,
        iq_data: &ArrayView3<Complex64>,
        depth: usize,
        beam: usize,
    ) -> (f64, f64) {
        let ensemble_size = iq_data.dim().0;

        // Compute lag-1 autocorrelation: R₁ = Σ(I_n * conj(I_{n+1}))
        let mut r1 = Complex64::new(0.0, 0.0);
        let mut power = 0.0;

        for n in 0..(ensemble_size - 1) {
            let sample_n = iq_data[[n, depth, beam]];
            let sample_n1 = iq_data[[n + 1, depth, beam]];

            r1 += sample_n * sample_n1.conj();
            power += sample_n.norm_sqr();
        }

        // Average autocorrelation
        let n_samples = (ensemble_size - 1) as f64;
        r1 /= n_samples;
        power /= n_samples;

        // Extract phase: φ = arctan(Im(R₁) / Re(R₁))
        let phase = r1.im.atan2(r1.re);

        // Convert phase to velocity
        let velocity = self.phase_to_velocity(phase);

        // Estimate variance from autocorrelation magnitude
        // Normalized variance: (1 - |R₁|/R₀)
        let r0 = power; // Lag-0 autocorrelation (signal power)
        let r1_magnitude = r1.norm();
        let normalized_variance = if r0 > 1e-10 {
            (1.0 - r1_magnitude / r0).max(0.0)
        } else {
            1.0 // High variance if no signal
        };

        (velocity, normalized_variance)
    }

    /// Convert Doppler phase shift to velocity
    fn phase_to_velocity(&self, phase: f64) -> f64 {
        // v = (φ * c) / (4π * f₀ * T_prf * cos(θ))
        let t_prf = 1.0 / self.config.prf;
        let cos_theta = self.config.beam_angle.cos();

        if cos_theta.abs() < 1e-6 {
            // Flow perpendicular to beam - no measurable velocity
            return 0.0;
        }

        (phase * self.config.speed_of_sound)
            / (4.0 * PI * self.config.center_frequency * t_prf * cos_theta)
    }

    /// Apply variance-based quality filtering
    ///
    /// Rejects velocity estimates with variance above threshold
    pub fn filter_by_variance(
        &self,
        velocity: &Array2<f64>,
        variance: &Array2<f64>,
    ) -> Array2<f64> {
        let mut filtered = velocity.clone();

        for ((i, j), var) in variance.indexed_iter() {
            if *var > self.config.variance_threshold {
                filtered[[i, j]] = 0.0; // Reject high-variance estimate
            }
        }

        filtered
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array3;

    #[test]
    fn test_autocorrelation_config() {
        let config = AutocorrelationConfig::default();

        // Check Nyquist velocity is reasonable
        let v_nyquist = config.nyquist_velocity();
        assert!(v_nyquist > 0.0, "Nyquist velocity should be positive");

        // Check velocity resolution
        let v_res = config.velocity_resolution();
        assert!(
            v_res > 0.0 && v_res < v_nyquist,
            "Velocity resolution should be positive and less than Nyquist"
        );
    }

    #[test]
    fn test_phase_to_velocity_conversion() {
        let config = AutocorrelationConfig {
            center_frequency: 5.0e6,
            prf: 4e3,
            speed_of_sound: 1540.0,
            beam_angle: 0.0, // cos(0) = 1
            ..Default::default()
        };

        let estimator = AutocorrelationEstimator::new(config);

        // Test zero phase → zero velocity
        let v = estimator.phase_to_velocity(0.0);
        assert_relative_eq!(v, 0.0, epsilon = 1e-9);

        // Test π/2 phase (should give specific velocity)
        let v_pi2 = estimator.phase_to_velocity(PI / 2.0);
        assert!(v_pi2 > 0.0, "Positive phase should give positive velocity");

        // Test -π/2 phase (should give negative velocity)
        let v_neg = estimator.phase_to_velocity(-PI / 2.0);
        assert!(v_neg < 0.0, "Negative phase should give negative velocity");
        assert_relative_eq!(v_neg, -v_pi2, epsilon = 1e-9);
    }

    #[test]
    fn test_estimate_zero_signal() {
        let config = AutocorrelationConfig::default();
        let estimator = AutocorrelationEstimator::new(config);

        // Create zero I/Q data
        let iq_data = Array3::<Complex64>::zeros((10, 16, 8));

        let result = estimator.estimate(&iq_data.view());
        assert!(result.is_ok(), "Zero signal should not error");

        let (velocity, variance) = result.unwrap();

        // Zero signal should give zero velocity
        assert!(
            velocity.iter().all(|&v| v == 0.0),
            "Zero signal should give zero velocity"
        );

        // Zero signal should give high variance (no correlation)
        assert!(
            variance.iter().all(|&var| var > 0.5),
            "Zero signal should give high variance"
        );
    }

    #[test]
    fn test_variance_filtering() {
        let config = AutocorrelationConfig {
            variance_threshold: 0.3,
            ..Default::default()
        };
        let estimator = AutocorrelationEstimator::new(config);

        let velocity = Array2::from_shape_fn((4, 4), |(i, j)| (i + j) as f64);
        let variance = Array2::from_shape_fn((4, 4), |(i, j)| {
            if i + j < 4 {
                0.2
            } else {
                0.5
            } // High variance in bottom-right
        });

        let filtered = estimator.filter_by_variance(&velocity, &variance);

        // Low-variance estimates should be preserved
        assert_eq!(filtered[[0, 0]], velocity[[0, 0]]);
        assert_eq!(filtered[[1, 1]], velocity[[1, 1]]);

        // High-variance estimates should be zeroed
        assert_eq!(filtered[[3, 3]], 0.0);
        assert_eq!(filtered[[2, 3]], 0.0);
    }
}
