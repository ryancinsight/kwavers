//! Data types and traits for uncertainty quantification.

use ndarray::Array3;

/// Uncertainty quantification methods.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UncertaintyMethod {
    /// Monte Carlo dropout sampling.
    MonteCarloDropout,
    /// Ensemble methods (bagging).
    Ensemble,
    /// Conformal prediction.
    Conformal,
    /// Sensitivity analysis.
    Sensitivity,
    /// Combined approach.
    Hybrid,
}

/// Configuration for uncertainty quantification.
#[derive(Debug, Clone)]
pub struct UncertaintyConfig {
    /// Primary uncertainty method.
    pub method: UncertaintyMethod,
    /// Number of uncertainty samples.
    pub num_samples: usize,
    /// Confidence level (0–1).
    pub confidence_level: f64,
    /// Dropout rate for MC dropout.
    pub dropout_rate: f64,
    /// Ensemble size for ensemble methods.
    pub ensemble_size: usize,
    /// Calibration set size for conformal prediction.
    pub calibration_size: usize,
}

impl Default for UncertaintyConfig {
    fn default() -> Self {
        Self {
            method: UncertaintyMethod::Ensemble,
            num_samples: 100,
            confidence_level: 0.95,
            dropout_rate: 0.1,
            ensemble_size: 10,
            calibration_size: 1000,
        }
    }
}

/// Common trait for uncertainty results.
pub trait UncertaintyResult: std::fmt::Debug {
    fn confidence_score(&self) -> f64;
    fn uncertainty_bounds(&self) -> (f64, f64);
}

/// Uncertainty result for beamforming.
#[derive(Debug)]
pub struct BeamformingUncertainty {
    pub uncertainty_map: Array3<f32>,
    pub confidence_score: f64,
    pub reliability_metrics: ReliabilityMetrics,
}

impl UncertaintyResult for BeamformingUncertainty {
    fn confidence_score(&self) -> f64 {
        self.confidence_score
    }

    fn uncertainty_bounds(&self) -> (f64, f64) {
        let min_uncertainty = self
            .uncertainty_map
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b as f64));
        let max_uncertainty = self
            .uncertainty_map
            .iter()
            .fold(0.0_f64, |a, &b| a.max(b as f64));
        (min_uncertainty, max_uncertainty)
    }
}

/// Reliability metrics for imaging quality.
#[derive(Debug)]
pub struct ReliabilityMetrics {
    pub signal_to_noise_ratio: f64,
    pub contrast_to_noise_ratio: f64,
    pub spatial_resolution: f64,
}

/// Uncertainty report summary.
#[derive(Debug)]
pub struct UncertaintySummary {
    pub mean_confidence: f64,
    pub confidence_range: (f64, f64),
    pub reliability_score: f64,
    pub dominant_uncertainty_sources: Vec<String>,
}

/// Complete uncertainty report.
#[derive(Debug)]
pub struct UncertaintyReport<'a> {
    pub summary: UncertaintySummary,
    pub detailed_results: Vec<&'a dyn UncertaintyResult>,
    pub recommendations: Vec<String>,
}
