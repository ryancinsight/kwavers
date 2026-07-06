//! Uncertainty Quantification Framework for Ultrasound Simulations.
//!
//! ## References
//!
//! - Kendall & Gal (2017): "What uncertainties do we need in Bayesian deep learning?"
//! - Angelopoulos & Bates (2021): "A Gentle Introduction to Conformal Prediction"
//! - Sullivan (2015): "Introduction to Uncertainty Quantification"

pub mod bayesian_networks;
pub mod conformal_prediction;
pub mod ensemble_methods;
pub mod predictor;
pub mod quantifier;
pub mod sensitivity_analysis;
#[cfg(test)]
mod tests;
pub mod types;

pub use bayesian_networks::{BayesianConfig, MlBayesianPINN, MlPredictionWithUncertainty};
pub use conformal_prediction::{ConformalConfig, ConformalResult, MlConformalPredictor};
pub use ensemble_methods::{EnsembleConfig, EnsembleQuantifier, EnsembleResult};
pub use predictor::PinnUncertaintyPredictor;
pub use quantifier::UncertaintyQuantifier;
pub use sensitivity_analysis::{SensitivityAnalyzer, SensitivityConfig, SensitivityIndices};
pub use types::{
    BeamformingUncertainty, MlUncertaintyConfig, MlUncertaintyMethod, ReliabilityMetrics,
    UncertaintyReport, UncertaintyResult, UncertaintySummary,
};
