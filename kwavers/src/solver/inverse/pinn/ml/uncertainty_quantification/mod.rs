//! Uncertainty Quantification for PINN Predictions.
//!
//! Implements Bayesian neural networks and uncertainty estimation techniques
//! for Physics-Informed Neural Networks.

pub mod bayesian;
pub mod conformal;
#[cfg(test)]
mod tests;
pub mod types;

pub use bayesian::PinnBayesianPINN;
pub use conformal::ConformalPredictor;
pub use types::{
    PinnUncertaintyConfig, PinnUncertaintyMethod, PredictionWithUncertainty, UncertaintyStats,
};
