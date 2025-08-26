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

// Re-export key types for easier access
pub use optimization::{
    AcousticEvent, CavitationEvent, ConvergencePredictor, ParameterOptimizer, PatternRecognizer,
    PatternSummary, SimulationPatterns,
};

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
    models: HashMap<ModelType, Model>,
    performance_metrics: PerformanceMetrics,
    _backend: MLBackend,
}

/// Type-safe model enum to avoid unsafe downcasts
#[derive(Debug)]
pub enum Model {
    /// Tissue classification model
    TissueClassifier(models::TissueClassifierModel),
    /// Parameter optimization model
    ParameterOptimizer(models::ParameterOptimizerModel),
    /// Anomaly detection model
    AnomalyDetector(models::AnomalyDetectorModel),
    /// Convergence prediction model
    ConvergencePredictor(models::ConvergencePredictorModel),
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

    /// Update model weights (for online learning)
    pub fn update(&mut self, gradients: &Array2<f32>) -> KwaversResult<()> {
        match self {
            Model::TissueClassifier(m) => m.update(gradients),
            Model::ParameterOptimizer(m) => m.update(gradients),
            Model::AnomalyDetector(m) => m.update(gradients),
            Model::ConvergencePredictor(m) => m.update(gradients),
        }
    }
}

/// Trait for ML models (now used by concrete model types)
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
    pub fn new(_backend: MLBackend) -> KwaversResult<Self> {
        let mut engine = Self {
            models: HashMap::new(),
            performance_metrics: PerformanceMetrics::default(),
            _backend,
        };

        // Initialize default models for Phase 12
        engine.initialize_default_models()?;

        Ok(engine)
    }

    /// Initialize default AI/ML models for Phase 12 capabilities
    fn initialize_default_models(&mut self) -> KwaversResult<()> {
        use crate::ml::models::TissueClassifierModel;

        // Initialize tissue classifier with enhanced features
        let tissue_classifier = TissueClassifierModel::with_random_weights(10, 5); // 10 features, 5 tissue types
        self.models.insert(
            ModelType::TissueClassifier,
            Model::TissueClassifier(tissue_classifier),
        );

        Ok(())
    }

    /// Load a pre-trained model
    ///
    /// Supports both ONNX models (recommended) and legacy JSON format for testing.
    /// ONNX models provide much better performance and are the standard format.
    pub fn load_model(&mut self, model_type: ModelType, path: &str) -> KwaversResult<()> {
        use std::path::Path;

        let path_obj = Path::new(path);
        let extension = path_obj
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");

        match extension {
            "onnx" => self.load_onnx_model(model_type, path),
            "json" => self.load_json_model(model_type, path),
            _ => Err(KwaversError::Data(crate::error::DataError::InvalidFormat {
                format: extension.to_string(),
                reason: "Unsupported model format. Use .onnx (recommended) or .json (testing)"
                    .to_string(),
            })),
        }
    }

    /// Load ONNX model (recommended format)
    #[cfg(feature = "ml")]
    fn load_onnx_model(&mut self, model_type: ModelType, path: &str) -> KwaversResult<()> {
        use ort::{Environment, SessionBuilder};

        // Create ONNX Runtime environment
        let environment = Environment::builder()
            .with_name("kwavers_ml")
            .build()
            .map_err(|e| {
                KwaversError::System(crate::error::SystemError::ExternalLibrary {
                    library: "onnx_runtime".to_string(),
                    error: e.to_string(),
                })
            })?;

        // Create session from ONNX model file
        let session = SessionBuilder::new(&environment)
            .map_err(|e| {
                KwaversError::System(crate::error::SystemError::ExternalLibrary {
                    library: "onnx_runtime".to_string(),
                    error: e.to_string(),
                })
            })?
            .with_model_from_file(path)
            .map_err(|e| {
                KwaversError::Data(crate::error::DataError::InvalidFormat {
                    format: "onnx".to_string(),
                    reason: e.to_string(),
                })
            })?;

        // Create model wrapper based on type
        let model = match model_type {
            ModelType::TissueClassifier => {
                let onnx_model = models::OnnxTissueClassifierModel::new(session)?;
                Model::TissueClassifier(models::TissueClassifierModel::Onnx(onnx_model))
            }
            ModelType::ParameterOptimizer => {
                let onnx_model = models::OnnxParameterOptimizerModel::new(session)?;
                Model::ParameterOptimizer(models::ParameterOptimizerModel::Onnx(onnx_model))
            }
            _ => {
                return Err(KwaversError::NotImplemented(format!(
                    "ONNX loading not yet implemented for {:?}",
                    model_type
                )));
            }
        };

        self.models.insert(model_type, model);
        log::info!("Loaded ONNX model: {} for {:?}", path, model_type);
        Ok(())
    }

    /// Load ONNX model (fallback when ml feature is disabled)
    #[cfg(not(feature = "ml"))]
    fn load_onnx_model(&mut self, _model_type: ModelType, _path: &str) -> KwaversResult<()> {
        Err(KwaversError::NotImplemented(
            "ONNX model loading requires the 'ml' feature to be enabled".to_string(),
        ))
    }

    /// Load JSON model format
    ///
    /// Note: ONNX format is preferred for production due to better performance.
    /// JSON format is useful for testing and debugging.
    fn load_json_model(&mut self, model_type: ModelType, path: &str) -> KwaversResult<()> {
        log::info!("Loading JSON model. Consider ONNX format for better performance.");

        // JSON loading implementation for testing and debugging
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
                let weights_json = json.get("weights").ok_or_else(|| {
                    KwaversError::Config(crate::error::ConfigError::MissingParameter {
                        parameter: "weights".to_string(),
                        section: "model".to_string(),
                    })
                })?;
                let weights_vec: Vec<Vec<f32>> = serde_json::from_value(weights_json.clone())
                    .map_err(|e| {
                        KwaversError::Data(crate::error::DataError::InvalidFormat {
                            format: "json".to_string(),
                            reason: e.to_string(),
                        })
                    })?;
                let rows = weights_vec.len();
                if rows == 0 {
                    return Err(KwaversError::Data(
                        crate::error::DataError::InsufficientData {
                            required: 1,
                            available: 0,
                        },
                    ));
                }
                let cols = weights_vec[0].len();
                if cols == 0 {
                    return Err(KwaversError::Data(
                        crate::error::DataError::InsufficientData {
                            required: 1,
                            available: 0,
                        },
                    ));
                }
                let flat: Vec<f32> = weights_vec.into_iter().flatten().collect();
                let weights = Array2::from_shape_vec((rows, cols), flat).map_err(|e| {
                    KwaversError::Data(crate::error::DataError::Corruption {
                        location: path.to_string(),
                        reason: e.to_string(),
                    })
                })?;

                let bias = if let Some(bias_json) = json.get("bias") {
                    let bias_vec: Vec<f32> =
                        serde_json::from_value(bias_json.clone()).map_err(|e| {
                            KwaversError::Data(crate::error::DataError::InvalidFormat {
                                format: "json".to_string(),
                                reason: e.to_string(),
                            })
                        })?;
                    Some(ndarray::Array1::from(bias_vec))
                } else {
                    None
                };

                let model = crate::ml::models::TissueClassifierModel::from_weights(weights, bias);
                self.models
                    .insert(model_type, Model::TissueClassifier(model));
            }
            _ => {
                return Err(KwaversError::NotImplemented(format!(
                    "JSON loading not supported for {:?}",
                    model_type
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
            .ok_or_else(|| {
                KwaversError::Config(crate::error::ConfigError::MissingParameter {
                    parameter: "TissueClassifier model".to_string(),
                    section: "MLEngine".to_string(),
                })
            })?;

        // Flatten the 3-D field into (cells, features=1)
        let flat_f32: Vec<f32> = field_data.iter().map(|&v| v as f32).collect();
        let cells = flat_f32.len();
        let input = Array2::from_shape_vec((cells, 1), flat_f32).map_err(|e| {
            KwaversError::System(crate::error::SystemError::MemoryAllocation {
                requested_bytes: cells * std::mem::size_of::<f32>(),
                reason: e.to_string(),
            })
        })?;

        // Type-safe inference using the Model enum - no unsafe downcasting needed
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
            .map_err(|e| {
                KwaversError::Data(crate::error::DataError::Corruption {
                    location: "classify_tissue reshape".to_string(),
                    reason: e.to_string(),
                })
            })?;

        // Update counters
        self.performance_metrics.total_inferences += 1;

        Ok(classes)
    }

    /// Run tissue classification **with uncertainty quantification** on the
    /// given 3-D field.  The method returns two volumes:
    /// 1. `classes` – the *arg-max* class IDs (*u8*) for each voxel.
    /// 2. `entropy` – the predictive entropy (*f32* in the range 0-ln(C))*.
    ///
    /// The predictive entropy is computed as *H(p) = -∑ p_i·ln(p_i)* where
    /// *p_i* are the soft-max probabilities output by the model.  Using
    /// entropy provides an intuitive measure of confidence that is easy to
    /// threshold for downstream tasks such as anomaly detection or adaptive
    /// refinement.
    pub fn classify_tissue_with_uncertainty(
        &mut self,
        field_data: &Array3<f64>,
    ) -> KwaversResult<(Array3<u8>, Array3<f32>)> {
        use ndarray::Array2;

        let model = self
            .models
            .get(&ModelType::TissueClassifier)
            .ok_or_else(|| {
                KwaversError::Config(crate::error::ConfigError::MissingParameter {
                    parameter: "TissueClassifier model".to_string(),
                    section: "MLEngine".to_string(),
                })
            })?;

        // Flatten 3-D field into (cells, features = 1)
        let flat_f32: Vec<f32> = field_data.mapv(|v| v as f32).into_raw_vec();
        let cells = flat_f32.len();
        let input = Array2::from_shape_vec((cells, 1), flat_f32).map_err(|e| {
            KwaversError::System(crate::error::SystemError::MemoryAllocation {
                requested_bytes: cells * std::mem::size_of::<f32>(),
                reason: e.to_string(),
            })
        })?;

        // Forward pass – obtain probability distribution for each voxel.
        let probs = model.infer(&input)?; // (cells, classes)
        let classes_count = probs.dim().1;
        if classes_count == 0 {
            return Err(KwaversError::Data(
                crate::error::DataError::InsufficientData {
                    required: 1,
                    available: 0,
                },
            ));
        }

        // Compute arg-max class and entropy per voxel.
        let mut classes_vec: Vec<u8> = Vec::with_capacity(cells);
        let mut entropy_vec: Vec<f32> = Vec::with_capacity(cells);

        for row in probs.rows() {
            // Arg-max
            let (idx, _max_p) =
                row.iter()
                    .enumerate()
                    .fold((0usize, f32::MIN), |(max_i, max_p), (i, &p)| {
                        if p > max_p {
                            (i, p)
                        } else {
                            (max_i, max_p)
                        }
                    });
            classes_vec.push(idx as u8);

            // Entropy
            let mut h = 0f32;
            for &p in row {
                if p > 0.0 {
                    h -= p * p.ln();
                }
            }
            entropy_vec.push(h);

            debug_assert!(h.is_finite(), "Entropy must be finite");
        }

        let classes = Array3::from_shape_vec(field_data.dim(), classes_vec).map_err(|e| {
            KwaversError::Data(crate::error::DataError::Corruption {
                location: "classify_tissue_with_uncertainty reshape".to_string(),
                reason: e.to_string(),
            })
        })?;
        let entropy = Array3::from_shape_vec(field_data.dim(), entropy_vec).map_err(|e| {
            KwaversError::Data(crate::error::DataError::Corruption {
                location: "classify_tissue_with_uncertainty entropy reshape".to_string(),
                reason: e.to_string(),
            })
        })?;

        // Update counters
        self.performance_metrics.total_inferences += 1;

        Ok((classes, entropy))
    }

    /// Optimize simulation parameters using the built-in RL optimiser.
    pub fn optimize_parameters(
        &self,
        current_params: &HashMap<String, f64>,
        target_metrics: &HashMap<String, f64>,
        simulation_state: &Array1<f64>,
    ) -> KwaversResult<HashMap<String, f64>> {
        use crate::ml::optimization::ParameterOptimizer;

        let mut optimizer = ParameterOptimizer::new(0.1, 0.05);
        optimizer.optimize_with_ai(current_params, target_metrics, simulation_state)
    }

    /// Detect anomalies via simple statistical thresholding (*mean + 3·σ*).
    pub fn detect_anomalies(&self, field_data: &Array3<f64>) -> KwaversResult<Vec<AnomalyRegion>> {
        let count = field_data.len() as f64;
        if count == 0.0 {
            return Ok(Vec::new());
        }

        let sum: f64 = field_data.iter().copied().sum();
        let mean = sum / count;
        let var_sum: f64 = field_data.iter().map(|v| (*v - mean).powi(2)).sum();
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
                    value: current_state.convergence_metric.to_string(),
                    min: "0.0".to_string(),
                    max: "∞".to_string(),
                },
            ));
        }

        // Very naive prediction: assume exponential decay and derive the time
        // to reach 1e-6 residual.
        let predicted = (current_state.convergence_metric / 1e-6_f64)
            .log10()
            .max(0.0);
        self.performance_metrics.total_inferences += 1;
        Ok(predicted)
    }

    /// Predict binary outcome probability using a ConvergencePredictor model.
    /// Returns *p_success* in range [0,1] for each sample in `features`.
    pub fn predict_outcome(&mut self, features: &Array2<f32>) -> KwaversResult<Array1<f32>> {
        use ndarray::Axis;
        let model = self
            .models
            .get(&ModelType::ConvergencePredictor)
            .ok_or_else(|| {
                KwaversError::Config(crate::error::ConfigError::MissingParameter {
                    parameter: "ConvergencePredictor model".to_string(),
                    section: "MLEngine".to_string(),
                })
            })?;
        let probs = model.infer(features)?;
        // Handle both single output (regression) and dual output (classification)
        let p_success = if probs.ncols() == 1 {
            // Single output - use sigmoid to convert to probability
            probs.column(0).mapv(|x| 1.0 / (1.0 + (-x).exp()))
        } else {
            // Multi-class - extract probability of success class
            probs.index_axis(Axis(1), 1).to_owned()
        };
        self.performance_metrics.total_inferences += probs.dim().0;
        Ok(p_success)
    }

    /// Get performance metrics
    pub fn performance_metrics(&self) -> &PerformanceMetrics {
        &self.performance_metrics
    }

    /// Parameter optimization using AI (Phase 12)
    pub fn optimize_parameters_ai(
        &self,
        current_params: &HashMap<String, f64>,
        target_params: &HashMap<String, f64>,
        simulation_state: &Array1<f64>,
        optimizer: &mut ParameterOptimizer,
    ) -> KwaversResult<HashMap<String, f64>> {
        optimizer.optimize_with_ai(current_params, target_params, simulation_state)
    }

    /// Pattern recognition for cavitation and acoustic events (Phase 12)
    pub fn analyze_simulation_patterns(
        &self,
        pressure: &Array3<f64>,
        bubble_radius: &Array3<f64>,
        acoustic_spectrum: &Array1<f64>,
        frequencies: &Array1<f64>,
    ) -> KwaversResult<SimulationPatterns> {
        let recognizer = PatternRecognizer::new();
        recognizer.analyze_simulation_patterns(
            pressure,
            bubble_radius,
            acoustic_spectrum,
            frequencies,
        )
    }

    /// Predict simulation convergence using AI (Phase 12)
    pub fn predict_convergence_ai(&self, residual_history: &[f64]) -> (bool, f64) {
        let predictor = ConvergencePredictor::new(20, 1e-6);
        predictor.predict_convergence(residual_history)
    }

    /// AI-assisted acceleration recommendations (Phase 12)
    pub fn suggest_acceleration_parameters(
        &self,
        current_residual: f64,
        residual_history: &[f64],
    ) -> KwaversResult<AIAccelerationRecommendation> {
        let predictor = ConvergencePredictor::new(20, 1e-6);

        // Calculate trend
        let trend = if residual_history.len() > 1 {
            let recent = &residual_history[residual_history.len().saturating_sub(10)..];
            let first_half = &recent[0..recent.len() / 2];
            let second_half = &recent[recent.len() / 2..];

            let avg_first: f64 = first_half.iter().sum::<f64>() / first_half.len() as f64;
            let avg_second: f64 = second_half.iter().sum::<f64>() / second_half.len() as f64;

            // Prevent division by zero
            if avg_first.abs() < 1e-15 {
                // If avg_first is effectively zero, use absolute difference as trend indicator
                if avg_second.abs() < 1e-15 {
                    0.0 // Both averages are zero - no trend
                } else {
                    -1.0 // Second half has values while first doesn't - trend down
                }
            } else {
                (avg_first - avg_second) / avg_first
            }
        } else {
            0.0
        };

        let acceleration_factor = predictor.suggest_acceleration(current_residual, trend);
        let (will_converge, confidence) = predictor.predict_convergence(residual_history);

        Ok(AIAccelerationRecommendation {
            acceleration_factor,
            confidence_level: confidence,
            predicted_convergence: will_converge,
            recommended_actions: self.generate_acceleration_actions(trend, current_residual),
        })
    }

    fn generate_acceleration_actions(&self, trend: f64, residual: f64) -> Vec<String> {
        let mut actions = Vec::new();

        if trend > 0.2 {
            actions.push("Increase time step size by 1.5x".to_string());
            actions.push("Enable multigrid acceleration".to_string());
        }

        if residual > 1e-3 {
            actions.push("Apply preconditioning".to_string());
            actions.push("Switch to adaptive time stepping".to_string());
        }

        if trend < 0.05 {
            actions.push("Reduce time step for stability".to_string());
            actions.push("Check for numerical instability".to_string());
        }

        if actions.is_empty() {
            actions.push("Continue with current parameters".to_string());
        }

        actions
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

/// AI-powered acceleration recommendations (Phase 12)
#[derive(Debug, Clone)]
pub struct AIAccelerationRecommendation {
    pub acceleration_factor: f64,
    pub confidence_level: f64,
    pub predicted_convergence: bool,
    pub recommended_actions: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ml_engine_creation() {
        let engine = MLEngine::new(MLBackend::Native).unwrap();
        // In Phase 12, engine now includes default models for AI capabilities
        assert!(!engine.models.is_empty()); // Should have default models
        assert!(engine.models.contains_key(&ModelType::TissueClassifier));
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

    #[test]
    fn test_classification_with_uncertainty() {
        use crate::ml::models::TissueClassifierModel;
        use ndarray::Array3;

        // Create a small 2×2×1 field with arbitrary values
        let field = Array3::from_shape_vec((2, 2, 1), vec![0.1, 0.2, 0.8, 0.9]).unwrap();

        // Build identity-like classifier: high positive weight for class-0 when value <0.5, class-1 otherwise
        // Since we only have one feature, we use opposite signs.
        let weights = ndarray::array![[5.0_f32, -5.0_f32]]; // (features=1, classes=2)
        let model = TissueClassifierModel::from_weights(weights, None);

        // Create engine and register model
        let mut engine = super::MLEngine::new(super::MLBackend::Native).unwrap();
        engine.models.insert(
            super::ModelType::TissueClassifier,
            Model::TissueClassifier(model),
        );

        // Run classification with uncertainty quantification
        let (classes, entropy) = engine.classify_tissue_with_uncertainty(&field).unwrap();

        // Shape checks
        assert_eq!(classes.dim(), field.dim());
        assert_eq!(entropy.dim(), field.dim());

        // Entropy must be non-negative and finite
        for &h in entropy.iter() {
            assert!(h.is_finite());
            assert!(h >= 0.0);
        }

        // Classes must be 0 or 1
        for &c in classes.iter() {
            assert!(c == 0 || c == 1);
        }
    }

    #[test]
    fn test_outcome_predictor() {
        use ndarray::{array, Array2};

        // Binary classifier with 2 outputs for softmax
        // First column: class 0 (failure), second column: class 1 (success)
        let weights = array![[-10.0_f32, 10.0_f32]]; // Feature positively correlates with success
        let bias = Some(array![5.0_f32, -5.0_f32]); // Bias towards failure for zero input
        let model = models::ConvergencePredictorModel::from_weights(weights, bias);

        let mut engine = MLEngine::new(MLBackend::Native).unwrap();
        engine.models.insert(
            ModelType::ConvergencePredictor,
            Model::ConvergencePredictor(model),
        );

        let features: Array2<f32> = array![[0.0], [1.0]];
        let probs = engine.predict_outcome(&features).unwrap();
        assert_eq!(probs.len(), 2);

        // With weights=10 and bias=-5:
        // For input 0.0: output = 0*10 - 5 = -5, sigmoid(-5) ≈ 0.007
        // For input 1.0: output = 1*10 - 5 = 5, sigmoid(5) ≈ 0.993
        assert!(
            probs[0] < 0.1,
            "Expected low probability for 0.0 input, got {}",
            probs[0]
        );
        assert!(
            probs[1] > 0.9,
            "Expected high probability for 1.0 input, got {}",
            probs[1]
        )
    }
}
