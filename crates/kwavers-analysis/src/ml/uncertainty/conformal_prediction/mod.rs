//! Conformal Prediction for Guaranteed Uncertainty Bounds
//!
//! Implements conformal prediction to provide distribution-free uncertainty
//! bounds with guaranteed coverage probabilities.

pub mod config;
pub mod predictor;
pub mod types;

pub use config::ConformalConfig;
pub use predictor::MlConformalPredictor;
pub use types::{
    CalibrationSummary, ConformalResult, ConformalValidationMetrics, PredictionIntervalBatch,
    ScoreDistribution,
};
