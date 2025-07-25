// Phase 12: AI/ML Integration Module
//! Machine learning and AI capabilities for intelligent simulation control
//! 
//! This module provides:
//! - Neural network inference for parameter optimization
//! - Pre-trained models for tissue classification
//! - Anomaly detection algorithms
//! - Reinforcement learning for automatic tuning

use crate::error::{KwaversError, KwaversResult};
use ndarray::{Array1, Array2, Array3};
use std::collections::HashMap;

pub mod inference;
pub mod models;
pub mod optimization;
pub mod training;

/// ML model types supported by the system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelType {
    /// Tissue classification model
    TissueClassifier,
    /// Parameter optimization model
    ParameterOptimizer,
    /// Anomaly detection model
    AnomalyDetector,
    /// Convergence prediction model
    ConvergencePredictor,
}

/// ML framework backend
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// Main ML engine for simulation intelligence
pub struct MLEngine {
    backend: MLBackend,
    models: HashMap<ModelType, Box<dyn MLModel>>,
    performance_metrics: PerformanceMetrics,
}

/// Trait for ML models
pub trait MLModel: Send + Sync {
    /// Get model type
    fn model_type(&self) -> ModelType;
    
    /// Run inference on input data
    fn infer(&self, input: &Array2<f32>) -> KwaversResult<Array2<f32>>;
    
    /// Get model metadata
    fn metadata(&self) -> &ModelMetadata;
    
    /// Update model weights (for online learning)
    fn update(&mut self, gradients: &Array2<f32>) -> KwaversResult<()>;
}

/// Model metadata
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub accuracy: f32,
    pub inference_time_ms: f32,
}

/// Performance metrics for ML operations
#[derive(Debug, Default)]
pub struct PerformanceMetrics {
    pub total_inferences: usize,
    pub average_inference_time_ms: f32,
    pub memory_usage_mb: f32,
    pub cache_hit_rate: f32,
}

impl MLEngine {
    /// Create new ML engine with specified backend
    pub fn new(backend: MLBackend) -> KwaversResult<Self> {
        Ok(Self {
            backend,
            models: HashMap::new(),
            performance_metrics: PerformanceMetrics::default(),
        })
    }
    
    /// Load a pre-trained model
    pub fn load_model(&mut self, model_type: ModelType, path: &str) -> KwaversResult<()> {
        // For the moment we support JSON-encoded weight matrices to avoid a
        // heavyweight ONNX dependency in unit tests.  The expected format is
        // `{ "weights": [[..],[..]], "bias": [..] }`.
        use std::fs;

        let content = fs::read_to_string(path).map_err(|_| {
            KwaversError::Data(crate::error::DataError::FileNotFound {
                path: path.to_string(),
            })
        })?;

        let json: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
            KwaversError::Data(crate::error::DataError::InvalidFormat {
                format: "json".to_string(),
                reason: e.to_string(),
            })
        })?;

        match model_type {
            ModelType::TissueClassifier => {
                use ndarray::Array2;
                let weights_json = json
                    .get("weights")
                    .ok_or_else(|| KwaversError::Config(crate::error::ConfigError::MissingParameter {
                        parameter: "weights".to_string(),
                        section: "model".to_string(),
                    }))?;
                let weights_vec: Vec<Vec<f32>> = serde_json::from_value(weights_json.clone())
                    .map_err(|e| KwaversError::Data(crate::error::DataError::InvalidFormat {
                        format: "json".to_string(),
                        reason: e.to_string(),
                    }))?;
                let rows = weights_vec.len();
                if rows == 0 {
                    return Err(KwaversError::Data(crate::error::DataError::InsufficientData {
                        required: 1,
                        available: 0,
                    }));
                }
                let cols = weights_vec[0].len();
                if cols == 0 {
                    return Err(KwaversError::Data(crate::error::DataError::InsufficientData {
                        required: 1,
                        available: 0,
                    }));
                }
                let flat: Vec<f32> = weights_vec.into_iter().flatten().collect();
                let weights = Array2::from_shape_vec((rows, cols), flat).map_err(|e| {
                    KwaversError::Data(crate::error::DataError::Corruption {
                        location: path.to_string(),
                        reason: e.to_string(),
                    })
                })?;

                let bias = if let Some(bias_json) = json.get("bias") {
                    let bias_vec: Vec<f32> = serde_json::from_value(bias_json.clone())
                        .map_err(|e| KwaversError::Data(crate::error::DataError::InvalidFormat {
                            format: "json".to_string(),
                            reason: e.to_string(),
                        }))?;
                    Some(ndarray::Array1::from(bias_vec))
                } else {
                    None
                };

                let model = crate::ml::models::TissueClassifierModel::from_weights(weights, bias);
                self.models.insert(model_type, Box::new(model));
            }
            _ => {
                return Err(KwaversError::NotImplemented(format!(
                    "Loading not supported for {:?}", model_type
                )));
            }
        }

        Ok(())
    }

    /// Run tissue classification on simulation data.  The method converts the
    /// *f64* field data to *f32*, feeds it through the tissue-classifier model
    /// and returns class IDs (arg-max) as an *u8* volume.
    pub fn classify_tissue(&mut self, field_data: &Array3<f64>) -> KwaversResult<Array3<u8>> {
        use ndarray::{Array2, Axis};

        let model = self
            .models
            .get(&ModelType::TissueClassifier)
            .ok_or_else(|| KwaversError::Config(crate::error::ConfigError::MissingParameter {
                parameter: "TissueClassifier model".to_string(),
                section: "MLEngine".to_string(),
            }))?;

        // Flatten the 3-D field into (cells, features=1)
        let flat_f32: Vec<f32> = field_data.iter().map(|&v| v as f32).collect();
        let cells = flat_f32.len();
        let input = Array2::from_shape_vec((cells, 1), flat_f32).map_err(|e| {
            KwaversError::System(crate::error::SystemError::MemoryAllocation {
                requested_bytes: cells * std::mem::size_of::<f32>(),
                reason: e.to_string(),
            })
        })?;

        // SAFETY: Downcast is safe because we only store TissueClassifierModel
        // under ModelType::TissueClassifier.
        let probs = model.infer(&input)?;
        let classes = probs
            .map_axis(Axis(1), |row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Less))
                    .map(|(idx, _)| idx as u8)
                    .unwrap_or(0)
            })
            .into_shape(field_data.dim())
            .map_err(|e| KwaversError::Data(crate::error::DataError::Corruption {
                location: "classify_tissue reshape".to_string(),
                reason: e.to_string(),
            }))?;

        // Update counters
        self.performance_metrics.total_inferences += 1;

        Ok(classes)
    }

    /// Optimize simulation parameters using the built-in RL optimiser.
    pub fn optimize_parameters(
        &self,
        current_params: &HashMap<String, f64>,
        target_metrics: &HashMap<String, f64>,
    ) -> KwaversResult<HashMap<String, f64>> {
        use crate::ml::optimization::ParameterOptimizer;

        let optimizer = ParameterOptimizer::new(0.1, 0.05);
        optimizer.optimize(current_params, target_metrics)
    }

    /// Detect anomalies via simple statistical thresholding (*mean + 3·σ*).
    pub fn detect_anomalies(&self, field_data: &Array3<f64>) -> KwaversResult<Vec<AnomalyRegion>> {
        let count = field_data.len() as f64;
        if count == 0.0 {
            return Ok(Vec::new());
        }

        let sum: f64 = field_data.iter().copied().sum();
        let mean = sum / count;
        let var_sum: f64 = field_data
            .iter()
            .map(|v| (*v - mean).powi(2))
            .sum();
        let std = (var_sum / count).sqrt();
        let threshold = mean + 3.0 * std;

        let mut anomalies = Vec::new();
        for ((x, y, z), &value) in field_data.indexed_iter() {
            if value > threshold {
                anomalies.push(AnomalyRegion {
                    center: [x, y, z],
                    radius: 1.0,
                    severity: ((value - mean) / std) as f32,
                    anomaly_type: "HighIntensity".to_string(),
                });
            }
        }

        Ok(anomalies)
    }

    /// Predict convergence as a simple reciprocal of the remaining error.
    pub fn predict_convergence(&mut self, current_state: &SimulationState) -> KwaversResult<f64> {
        if current_state.convergence_metric <= 0.0 {
            return Err(KwaversError::Validation(
                crate::error::ValidationError::RangeValidation {
                    field: "convergence_metric".to_string(),
                    value: current_state.convergence_metric,
                    min: 0.0,
                    max: f64::INFINITY,
                },
            ));
        }

        // Very naive prediction: assume exponential decay and derive the time
        // to reach 1e-6 residual.
        let predicted = (current_state.convergence_metric / 1e-6_f64).log10().max(0.0);
        self.performance_metrics.total_inferences += 1;
        Ok(predicted)
    }
}

/// Represents an anomalous region in the simulation
#[derive(Debug, Clone)]
pub struct AnomalyRegion {
    pub center: [usize; 3],
    pub radius: f64,
    pub severity: f32,
    pub anomaly_type: String,
}

/// Current simulation state for ML analysis
#[derive(Debug, Clone)]
pub struct SimulationState {
    pub timestep: usize,
    pub max_pressure: f64,
    pub total_energy: f64,
    pub convergence_metric: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ml_engine_creation() {
        let engine = MLEngine::new(MLBackend::Native).unwrap();
        assert_eq!(engine.backend, MLBackend::Native);
        assert!(engine.models.is_empty());
    }
    
    #[test]
    fn test_model_type_enum() {
        let model_type = ModelType::TissueClassifier;
        assert_eq!(model_type, ModelType::TissueClassifier);
    }
    
    #[test]
    fn test_ml_backend_enum() {
        let backend = MLBackend::ONNX;
        assert_eq!(backend, MLBackend::ONNX);
    }
}