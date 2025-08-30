//! Machine learning models for acoustic simulation
//!
//! This module provides ML models for various simulation tasks.
//! Each model is in its own submodule for proper separation of concerns.

pub mod anomaly_detector;
pub mod convergence_predictor;
pub mod outcome_predictor;
pub mod parameter_optimizer;
pub mod tissue_classifier;

// Re-export main types
pub use anomaly_detector::AnomalyDetectorModel;
pub use convergence_predictor::ConvergencePredictorModel;
pub use outcome_predictor::OutcomePredictorModel;
pub use parameter_optimizer::ParameterOptimizerModel;
pub use tissue_classifier::TissueClassifierModel;

use crate::error::KwaversResult;
use ndarray::Array2;

// Re-export MLModel and ModelMetadata from types
pub use crate::ml::types::{MLModel, ModelMetadata};

/// Model type enumeration
#[derive(Debug, Clone, Copy)]
pub enum ModelType {
    TissueClassifier,
    ParameterOptimizer,
    AnomalyDetector,
    ConvergencePredictor,
    OutcomePredictor,
}

/// ML constants
pub mod constants {
    /// Default batch size for inference
    pub const DEFAULT_BATCH_SIZE: usize = 32;

    /// Maximum sequence length for temporal models
    pub const MAX_SEQUENCE_LENGTH: usize = 1000;

    /// Default learning rate
    pub const DEFAULT_LEARNING_RATE: f32 = 0.001;

    /// Convergence threshold
    pub const CONVERGENCE_THRESHOLD: f32 = 1e-6;
}
