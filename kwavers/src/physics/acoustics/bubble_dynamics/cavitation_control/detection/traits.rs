//! Cavitation detector trait definition

use crate::core::constants::numerical::MHZ_TO_HZ;
use super::types::CavitationMetrics;
use ndarray::ArrayView1;

/// Trait for cavitation detection algorithms
pub trait CavitationDetector: Send + Sync {
    /// Detect cavitation in the given signal
    fn detect(&mut self, signal: &ArrayView1<f64>) -> CavitationMetrics;

    /// Reset detector state
    fn reset(&mut self);

    /// Get detection method
    fn method(&self) -> super::types::DetectionMethod;

    /// Update detector parameters
    fn update_parameters(&mut self, params: DetectorParameters);
}

/// Parameters for cavitation detectors
#[derive(Debug, Clone)]
pub struct DetectorParameters {
    /// Fundamental (drive) frequency \[Hz\].
    pub fundamental_freq: f64,
    /// Sampling rate for `signal` \[Hz\].
    pub sample_rate: f64,
    /// Dimensionless scaling applied to detector thresholds.
    pub sensitivity: f64,
    /// Enable temporal averaging of detector outputs when supported.
    pub temporal_averaging: bool,
    /// Enable adaptive baseline/thresholding when supported.
    pub adaptive_threshold: bool,
}

impl Default for DetectorParameters {
    fn default() -> Self {
        Self {
            fundamental_freq: MHZ_TO_HZ,       // 1 MHz
            sample_rate: 10.0 * MHZ_TO_HZ,     // 10 MHz
            sensitivity: 1.0,
            temporal_averaging: true,
            adaptive_threshold: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Default DetectorParameters satisfies Nyquist: sample_rate > 2 * fundamental_freq.
    #[test]
    fn default_detector_parameters_nyquist_satisfied() {
        let p = DetectorParameters::default();
        assert!(
            p.sample_rate > 2.0 * p.fundamental_freq,
            "sample_rate ({}) must exceed 2·fundamental_freq ({})",
            p.sample_rate,
            p.fundamental_freq
        );
    }

    /// Default sensitivity is 1.0 and temporal_averaging is true.
    #[test]
    fn default_detector_parameters_sensitivity_and_flags() {
        let p = DetectorParameters::default();
        assert!((p.sensitivity - 1.0).abs() < 1e-15);
        assert!(p.temporal_averaging);
        assert!(!p.adaptive_threshold);
    }

    /// Clone produces a copy with identical field values.
    #[test]
    fn detector_parameters_clone_equal() {
        let p = DetectorParameters::default();
        let c = p.clone();
        assert!((p.fundamental_freq - c.fundamental_freq).abs() < 1e-15);
        assert!((p.sample_rate - c.sample_rate).abs() < 1e-15);
        assert!((p.sensitivity - c.sensitivity).abs() < 1e-15);
    }
}
