//! ML Engine implementation
//!
//! This module contains the main MLEngine struct and its implementation
//! for managing machine learning models and inference.

use crate::error::{ConfigError, DataError, KwaversError, KwaversResult, SystemError};
use ndarray::{Array1, Array2, Array3};
use std::collections::HashMap;

use super::{
    models::{
        AnomalyDetectorModel, ConvergencePredictorModel, ParameterOptimizerModel,
        TissueClassifierModel,
    },
    MLBackend, Model, ModelType, PerformanceMetrics,
};

/// Main ML engine for simulation intelligence
#[derive(Debug)]
pub struct MLEngine {
    pub(crate) models: HashMap<ModelType, Model>,
    pub(crate) performance_metrics: PerformanceMetrics,
    pub(crate) backend: MLBackend,
}

impl MLEngine {
    /// Create new ML engine with specified backend
    pub fn new(backend: MLBackend) -> KwaversResult<Self> {
        Ok(Self {
            models: HashMap::new(),
            performance_metrics: PerformanceMetrics::default(),
            backend,
        })
    }

    /// Load a model from file
    pub fn load_model(&mut self, model_type: ModelType, path: &str) -> KwaversResult<()> {
        use crate::ml::types::MLModel;
        use std::path::Path;

        let path = Path::new(path);
        let model = match model_type {
            ModelType::TissueClassifier => {
                Model::TissueClassifier(TissueClassifierModel::load(path)?)
            }
            ModelType::ParameterOptimizer => {
                Model::ParameterOptimizer(ParameterOptimizerModel::load(path)?)
            }
            ModelType::AnomalyDetector => Model::AnomalyDetector(AnomalyDetectorModel::load(path)?),
            ModelType::ConvergencePredictor => {
                Model::ConvergencePredictor(ConvergencePredictorModel::load(path)?)
            }
            ModelType::OutcomePredictor => {
                // OutcomePredictor not yet implemented in Model enum
                return Err(crate::error::KwaversError::NotImplemented(
                    "OutcomePredictor model loading not yet implemented".to_string(),
                ));
            }
        };

        self.models.insert(model_type, model);
        Ok(())
    }

    /// Run tissue classification on 3D field
    pub fn classify_tissue(&mut self, field_data: &Array3<f64>) -> KwaversResult<Array3<u8>> {
        let model = self
            .models
            .get(&ModelType::TissueClassifier)
            .ok_or_else(|| {
                KwaversError::Config(ConfigError::MissingParameter {
                    parameter: "TissueClassifier model".to_string(),
                    section: "MLEngine".to_string(),
                })
            })?;

        // Flatten 3D field to 2D for inference
        let (flat_f32, _offset) = field_data.mapv(|v| v as f32).into_raw_vec_and_offset();
        let cells = flat_f32.len();
        let input = Array2::from_shape_vec((cells, 1), flat_f32).map_err(|e| {
            KwaversError::System(SystemError::MemoryAllocation {
                requested_bytes: cells * std::mem::size_of::<f32>(),
                reason: e.to_string(),
            })
        })?;

        // Run inference
        let output = model.infer(&input)?;

        // Convert probabilities to class labels
        let classes: Vec<u8> = output
            .rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as u8)
                    .unwrap_or(0)
            })
            .collect();

        // Reshape back to 3D
        let classes = Array3::from_shape_vec(field_data.dim(), classes).map_err(|e| {
            KwaversError::Data(DataError::Corruption {
                location: "classify_tissue reshape".to_string(),
                reason: e.to_string(),
            })
        })?;

        self.performance_metrics.total_inferences += 1;
        Ok(classes)
    }

    /// Run tissue classification with uncertainty quantification
    pub fn classify_tissue_with_uncertainty(
        &mut self,
        field_data: &Array3<f64>,
    ) -> KwaversResult<(Array3<u8>, Array3<f32>)> {
        let model = self
            .models
            .get(&ModelType::TissueClassifier)
            .ok_or_else(|| {
                KwaversError::Config(ConfigError::MissingParameter {
                    parameter: "TissueClassifier model".to_string(),
                    section: "MLEngine".to_string(),
                })
            })?;

        // Flatten 3D field into (cells, features = 1)
        let (flat_f32, _offset) = field_data.mapv(|v| v as f32).into_raw_vec_and_offset();
        let cells = flat_f32.len();
        let input = Array2::from_shape_vec((cells, 1), flat_f32).map_err(|e| {
            KwaversError::System(SystemError::MemoryAllocation {
                requested_bytes: cells * std::mem::size_of::<f32>(),
                reason: e.to_string(),
            })
        })?;

        // Forward pass – obtain probability distribution for each voxel
        let probs = model.infer(&input)?;
        let classes_count = probs.dim().1;
        if classes_count == 0 {
            return Err(KwaversError::Data(DataError::InsufficientData {
                required: 1,
                available: 0,
            }));
        }

        // Compute arg-max class and entropy per voxel
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

            // Entropy: H(p) = -∑ p_i·ln(p_i)
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
            KwaversError::Data(DataError::Corruption {
                location: "classify_tissue_with_uncertainty reshape".to_string(),
                reason: e.to_string(),
            })
        })?;

        let entropy = Array3::from_shape_vec(field_data.dim(), entropy_vec).map_err(|e| {
            KwaversError::Data(DataError::Corruption {
                location: "classify_tissue_with_uncertainty entropy reshape".to_string(),
                reason: e.to_string(),
            })
        })?;

        self.performance_metrics.total_inferences += 1;
        Ok((classes, entropy))
    }

    /// Optimize simulation parameters
    pub fn optimize_parameters(
        &mut self,
        current_state: &Array2<f32>,
    ) -> KwaversResult<Array1<f32>> {
        let model = self
            .models
            .get(&ModelType::ParameterOptimizer)
            .ok_or_else(|| {
                KwaversError::Config(ConfigError::MissingParameter {
                    parameter: "ParameterOptimizer model".to_string(),
                    section: "MLEngine".to_string(),
                })
            })?;

        let output = model.infer(current_state)?;

        // Extract first row as parameter adjustments
        let params = output.row(0).to_owned();

        self.performance_metrics.total_optimizations += 1;
        Ok(params)
    }

    /// Detect anomalies in simulation
    pub fn detect_anomalies(&mut self, field_data: &Array3<f64>) -> KwaversResult<Array3<bool>> {
        let model = self
            .models
            .get(&ModelType::AnomalyDetector)
            .ok_or_else(|| {
                KwaversError::Config(ConfigError::MissingParameter {
                    parameter: "AnomalyDetector model".to_string(),
                    section: "MLEngine".to_string(),
                })
            })?;

        // Flatten and convert
        let (flat_f32, _offset) = field_data.mapv(|v| v as f32).into_raw_vec_and_offset();
        let cells = flat_f32.len();
        let input = Array2::from_shape_vec((cells, 1), flat_f32).map_err(|e| {
            KwaversError::System(SystemError::MemoryAllocation {
                requested_bytes: cells * std::mem::size_of::<f32>(),
                reason: e.to_string(),
            })
        })?;

        let output = model.infer(&input)?;

        // Threshold at 0.5 for binary classification
        let anomalies: Vec<bool> = output.column(0).iter().map(|&v| v > 0.5).collect();

        let anomalies = Array3::from_shape_vec(field_data.dim(), anomalies).map_err(|e| {
            KwaversError::Data(DataError::Corruption {
                location: "detect_anomalies reshape".to_string(),
                reason: e.to_string(),
            })
        })?;

        self.performance_metrics.total_anomalies_detected +=
            anomalies.iter().filter(|&&a| a).count();
        Ok(anomalies)
    }

    /// Predict convergence probability
    pub fn predict_convergence(&mut self, features: &Array2<f32>) -> KwaversResult<f32> {
        let model = self
            .models
            .get(&ModelType::ConvergencePredictor)
            .ok_or_else(|| {
                KwaversError::Config(ConfigError::MissingParameter {
                    parameter: "ConvergencePredictor model".to_string(),
                    section: "MLEngine".to_string(),
                })
            })?;

        let output = model.infer(features)?;

        // Return probability of convergence (assuming binary classification)
        Ok(output[[0, 1]])
    }

    /// Predict outcome probabilities for multiple samples
    pub fn predict_outcome(&mut self, features: &Array2<f32>) -> KwaversResult<Vec<f32>> {
        let model = self
            .models
            .get(&ModelType::ConvergencePredictor)
            .ok_or_else(|| {
                KwaversError::Config(ConfigError::MissingParameter {
                    parameter: "ConvergencePredictor model".to_string(),
                    section: "MLEngine".to_string(),
                })
            })?;

        let output = model.infer(features)?;

        // Extract success probability for each sample
        let probs: Vec<f32> = output.column(1).to_vec();
        Ok(probs)
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> &PerformanceMetrics {
        &self.performance_metrics
    }

    /// Reset performance metrics
    pub fn reset_metrics(&mut self) {
        self.performance_metrics = PerformanceMetrics::default();
    }
}
