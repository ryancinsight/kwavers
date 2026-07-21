//! Uncertainty Quantification for PINN Predictions.
//!
//! Implements Bayesian neural networks and uncertainty estimation techniques
//! for Physics-Informed Neural Networks.

pub mod bayesian;
pub mod conformal;
mod precision;
mod statistics;
#[cfg(test)]
mod tests;
pub mod types;

pub use bayesian::PinnBayesianPINN;
pub use conformal::PinnConformalPredictor;
pub use types::{
    PinnPredictionWithUncertainty, PinnUncertaintyConfig, PinnUncertaintyMethod, UncertaintyStats,
};
