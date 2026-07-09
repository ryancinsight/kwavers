//! Bayesian Neural Networks for Uncertainty Quantification
//!
//! Implements Bayesian neural networks using Monte Carlo dropout for
//! uncertainty estimation in physics-informed neural networks.

mod calibration;
mod inference;
#[cfg(test)]
mod tests;

use leto::Array2;
use std::collections::HashMap;

/// Configuration for Bayesian PINN
#[derive(Debug, Clone)]
pub struct BayesianConfig {
    /// Dropout rate for uncertainty estimation
    pub dropout_rate: f64,
    /// Number of Monte Carlo samples
    pub num_samples: usize,
}

impl Default for BayesianConfig {
    fn default() -> Self {
        Self {
            dropout_rate: 0.1,
            num_samples: 100,
        }
    }
}

/// Prediction with uncertainty bounds
#[derive(Debug, Clone)]
pub struct MlPredictionWithUncertainty {
    /// Mean prediction across samples
    pub mean_prediction: Array2<f32>,
    /// Uncertainty (standard deviation)
    pub uncertainty: Array2<f32>,
    /// Confidence intervals for different levels
    pub confidence_intervals: HashMap<String, (Array2<f32>, Array2<f32>)>,
    /// Overall reliability score
    pub reliability_score: f64,
}

/// Uncertainty decomposition into epistemic and aleatoric components
#[derive(Debug)]
pub struct UncertaintyDecomposition {
    /// Total uncertainty (epistemic + aleatoric)
    pub total_uncertainty: Array2<f32>,
    /// Epistemic uncertainty (model uncertainty)
    pub epistemic_uncertainty: Array2<f32>,
    /// Aleatoric uncertainty (data noise)
    pub aleatoric_uncertainty: Array2<f32>,
    /// Ratio of epistemic to aleatoric uncertainty
    pub uncertainty_ratio: f64,
}

/// Bayesian Physics-Informed Neural Network.
///
/// MC dropout masks are generated on-the-fly during forward passes rather than
/// stored as a field.  The `BayesianConfig` drives the sampling parameters.
#[derive(Debug)]
pub struct MlBayesianPINN {
    pub(self) _config: BayesianConfig,
}

impl MlBayesianPINN {
    /// Create new Bayesian PINN.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(config: BayesianConfig) -> kwavers_core::error::KwaversResult<Self> {
        Ok(Self { _config: config })
    }
}
