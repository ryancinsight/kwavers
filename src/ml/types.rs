//! ML type definitions and enums
//!
//! This module contains the core type definitions for the ML subsystem.

use super::models::{
    AnomalyDetectorModel, ConvergencePredictorModel, ParameterOptimizerModel, TissueClassifierModel,
};
use crate::error::KwaversResult;
use ndarray::Array2;

/// ML model types supported by the system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash))]
pub enum ModelType {
    /// Tissue classification model
    TissueClassifier,
    /// Parameter optimization model
    ParameterOptimizer,
    /// Anomaly detection model
    AnomalyDetector,
    /// Convergence prediction model
    ConvergencePredictor,
    /// Outcome prediction model
    OutcomePredictor,
}

/// ML framework backend
#[derive(Debug, Clone, Copy, PartialEq, Eq))]
pub enum MLBackend {
    /// ONNX Runtime for cross-platform inference
    ONNX,
    /// TensorFlow Lite for mobile/embedded
    TFLite,
    /// PyTorch Mobile
    PyTorchMobile,
    /// Custom Rust implementation
    Native,
}

/// Type-safe model enum to avoid unsafe downcasts
#[derive(Debug))]
pub enum Model {
    /// Tissue classification model
    TissueClassifier(TissueClassifierModel),
    /// Parameter optimization model
    ParameterOptimizer(ParameterOptimizerModel),
    /// Anomaly detection model
    AnomalyDetector(AnomalyDetectorModel),
    /// Convergence prediction model
    ConvergencePredictor(ConvergencePredictorModel),
}

impl Model {
    /// Get model type
    pub fn model_type(&self) -> ModelType {
        match self {
            Model::TissueClassifier(_) => ModelType::TissueClassifier,
            Model::ParameterOptimizer(_) => ModelType::ParameterOptimizer,
            Model::AnomalyDetector(_) => ModelType::AnomalyDetector,
            Model::ConvergencePredictor(_) => ModelType::ConvergencePredictor,
        }
    }

    /// Run inference on input data
    pub fn infer(&self, input: &Array2<f32>) -> KwaversResult<Array2<f32>> {
        match self {
            Model::TissueClassifier(m) => m.infer(input),
            Model::ParameterOptimizer(m) => m.infer(input),
            Model::AnomalyDetector(m) => m.infer(input),
            Model::ConvergencePredictor(m) => m.infer(input),
        }
    }

    /// Get model metadata
    pub fn metadata(&self) -> &ModelMetadata {
        match self {
            Model::TissueClassifier(m) => m.metadata(),
            Model::ParameterOptimizer(m) => m.metadata(),
            Model::AnomalyDetector(m) => m.metadata(),
            Model::ConvergencePredictor(m) => m.metadata(),
        }
    }
}

/// Performance metrics for ML operations
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    pub total_inferences: usize,
    pub total_optimizations: usize,
    pub total_anomalies_detected: usize,
    pub average_inference_time_ms: f64,
    pub peak_memory_usage_mb: f64,
}

/// Configuration for ML models
#[derive(Debug, Clone))]
pub struct MLConfig {
    pub backend: MLBackend,
    pub model_cache_size_mb: usize,
    pub batch_size: usize,
    pub num_threads: usize,
    pub enable_gpu: bool,
    pub precision: InferencePrecision,
}

impl Default for MLConfig {
    fn default() -> Self {
        Self {
            backend: MLBackend::Native,
            model_cache_size_mb: 100,
            batch_size: 32,
            num_threads: 4,
            enable_gpu: false,
            precision: InferencePrecision::Float32,
        }
    }
}

/// Inference precision mode
#[derive(Debug, Clone, Copy, PartialEq, Eq))]
pub enum InferencePrecision {
    /// Full 32-bit floating point
    Float32,
    /// Half precision (16-bit)
    Float16,
    /// 8-bit integer quantization
    Int8,
    /// Mixed precision
    Mixed,
}

/// Model metadata information
#[derive(Debug, Clone))]
pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub accuracy: f64,
    pub inference_time_ms: f64,
}

/// Trait for ML models
pub trait MLModel {
    /// Get model metadata
    fn metadata(&self) -> &ModelMetadata;

    /// Get model type
    fn model_type(&self) -> ModelType;

    /// Run inference (alias for predict)
    fn infer(&self, input: &Array2<f32>) -> KwaversResult<Array2<f32>>;

    /// Run inference on input data
    fn predict(&self, input: &Array2<f32>) -> KwaversResult<Array2<f32>> {
        self.infer(input)
    }

    /// Get model accuracy metric
    fn accuracy(&self) -> f64;

    /// Get model name
    fn name(&self) -> &str {
        self.metadata().name.as_str()
    }

    /// Update model weights with gradients
    fn update(&mut self, gradients: &Array2<f32>) -> KwaversResult<()>;

    /// Load model from file
    fn load(path: &str) -> KwaversResult<Self>
    where
        Self: Sized;

    /// Save model to file
    fn save(&self, path: &str) -> KwaversResult<()>;
}
