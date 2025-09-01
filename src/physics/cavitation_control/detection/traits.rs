//! Cavitation detector trait definition

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
    pub fundamental_freq: f64,
    pub sample_rate: f64,
    pub sensitivity: f64,
    pub temporal_averaging: bool,
    pub adaptive_threshold: bool,
}

impl Default for DetectorParameters {
    fn default() -> Self {
        Self {
            fundamental_freq: 1e6, // 1 MHz
            sample_rate: 10e6,     // 10 MHz
            sensitivity: 1.0,
            temporal_averaging: true,
            adaptive_threshold: false,
        }
    }
}
