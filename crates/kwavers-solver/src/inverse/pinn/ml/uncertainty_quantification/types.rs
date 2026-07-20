//! PINN uncertainty quantification types.

/// Uncertainty quantification configuration.
#[derive(Debug, Clone)]
pub struct PinnUncertaintyConfig {
    /// Number of Monte Carlo samples for dropout uncertainty.
    pub mc_samples: usize,
    /// Dropout probability for MC dropout.
    pub dropout_prob: f64,
    /// Ensemble size for deep ensembles.
    pub ensemble_size: usize,
    /// Conformal prediction alpha (1-confidence level).
    pub conformal_alpha: f32,
    /// Variance threshold for reliability assessment.
    pub variance_threshold: f32,
}

/// Prediction with uncertainty bounds.
#[derive(Debug, Clone)]
pub struct PinnPredictionWithUncertainty {
    /// Mean prediction.
    pub mean: Vec<f32>,
    /// Standard deviation.
    pub std: Vec<f32>,
    /// 95% confidence interval (lower, upper).
    pub confidence_interval: (Vec<f32>, Vec<f32>),
    /// Predictive entropy (measure of uncertainty).
    pub entropy: f32,
    /// Reliability score (0-1, higher is more reliable).
    pub reliability: f32,
    /// Uncertainty quantification method used.
    pub method: PinnUncertaintyMethod,
}

/// Uncertainty estimation methods.
#[derive(Debug, Clone)]
pub enum PinnUncertaintyMethod {
    /// Monte Carlo Dropout.
    MCDropout,
    /// Deep Ensemble.
    DeepEnsemble,
    /// Conformal Prediction.
    Conformal,
    /// Combined methods.
    Hybrid,
}

/// Uncertainty estimation statistics.
#[derive(Debug, Clone)]
pub struct UncertaintyStats {
    pub total_predictions: usize,
    pub average_uncertainty: f32,
    pub calibration_error: f32,
    pub coverage_probability: f32,
    pub reliability_score: f32,
}

impl Default for UncertaintyStats {
    fn default() -> Self {
        Self {
            total_predictions: 0,
            average_uncertainty: 0.0,
            calibration_error: 0.0,
            coverage_probability: 0.0,
            reliability_score: 0.0,
        }
    }
}
