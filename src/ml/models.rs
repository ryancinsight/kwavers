//! Pre-trained models for tissue classification and parameter optimization

use crate::error::{KwaversError, KwaversResult};
use crate::ml::{inference::InferenceEngine, MLModel, ModelMetadata, ModelType};
use ndarray::{Array1, Array2, Axis};
use rand::Rng;

/// Tissue classification model
#[derive(Debug)]
pub struct TissueClassifierModel {
    pub(crate) engine: InferenceEngine,
    metadata: ModelMetadata,
}

impl TissueClassifierModel {
    /// Create a *randomly initialised* classifier with the given dimensionality.
    pub fn with_random_weights(features: usize, classes: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = Array2::from_shape_fn((features, classes), |_| rng.gen_range(-0.05..0.05));
        let engine = InferenceEngine::from_weights(weights, None, 32, false);

        let metadata = ModelMetadata {
            name: "TissueClassifier".to_string(),
            version: "0.1.0".to_string(),
            input_shape: vec![features],
            output_shape: vec![classes],
            accuracy: 0.0,
            inference_time_ms: 0.0,
        };

        Self { engine, metadata }
    }

    /// Create a classifier from explicit weights / bias.
    pub fn from_weights(weights: Array2<f32>, bias: Option<Array1<f32>>) -> Self {
        let (features, classes) = weights.dim();
        let engine = InferenceEngine::from_weights(weights, bias, 32, false);

        let metadata = ModelMetadata {
            name: "TissueClassifier".to_string(),
            version: "1.0.0".to_string(),
            input_shape: vec![features],
            output_shape: vec![classes],
            accuracy: 0.0,
            inference_time_ms: 0.0,
        };

        Self { engine, metadata }
    }
}

impl MLModel for TissueClassifierModel {
    fn model_type(&self) -> ModelType {
        ModelType::TissueClassifier
    }

    fn infer(&self, input: &Array2<f32>) -> KwaversResult<Array2<f32>> {
        // Convert 2D input to 3D for the engine
        let (batch, features) = input.dim();
        let input_3d = input
            .clone()
            .into_shape((batch, features, 1))
            .map_err(|e| {
                KwaversError::System(crate::error::SystemError::MemoryAllocation {
                    requested_bytes: batch * features * std::mem::size_of::<f32>(),
                    reason: e.to_string(),
                })
            })?;

        // Run inference
        let output_3d = self.engine.infer_batch(&input_3d)?;

        // Convert back to 2D
        let output_2d = output_3d.index_axis(Axis(2), 0).to_owned();
        Ok(output_2d)
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn update(&mut self, gradients: &Array2<f32>) -> KwaversResult<()> {
        // Gradient descent update
        let learning_rate = 1e-3_f32;

        // Sanity-check dimensionality
        if gradients.shape() != self.engine.weights_mut().shape() {
            return Err(KwaversError::Physics(
                crate::error::PhysicsError::DimensionMismatch,
            ));
        }

        let weights = self.engine.weights_mut();
        *weights -= &(gradients * learning_rate);
        Ok(())
    }
}

/// Parameter optimization model
#[derive(Debug)]
pub struct ParameterOptimizerModel {
    pub(crate) engine: InferenceEngine,
    metadata: ModelMetadata,
}

impl ParameterOptimizerModel {
    /// Create a randomly initialized parameter optimizer
    pub fn with_random_weights(features: usize, outputs: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = Array2::from_shape_fn((features, outputs), |_| rng.gen_range(-0.1..0.1));
        let engine = InferenceEngine::from_weights(weights, None, 64, false);

        let metadata = ModelMetadata {
            name: "ParameterOptimizer".to_string(),
            version: "1.0.0".to_string(),
            input_shape: vec![features],
            output_shape: vec![outputs],
            accuracy: 0.0,
            inference_time_ms: 0.0,
        };

        Self { engine, metadata }
    }

    /// Create from explicit weights and bias
    pub fn from_weights(weights: Array2<f32>, bias: Option<Array1<f32>>) -> Self {
        let (features, outputs) = weights.dim();
        let engine = InferenceEngine::from_weights(weights, bias, 64, false);

        let metadata = ModelMetadata {
            name: "ParameterOptimizer".to_string(),
            version: "1.0.0".to_string(),
            input_shape: vec![features],
            output_shape: vec![outputs],
            accuracy: 0.0,
            inference_time_ms: 0.0,
        };

        Self { engine, metadata }
    }
}

impl MLModel for ParameterOptimizerModel {
    fn model_type(&self) -> ModelType {
        ModelType::ParameterOptimizer
    }

    fn infer(&self, input: &Array2<f32>) -> KwaversResult<Array2<f32>> {
        // Convert 2D input to 3D for the engine
        let (batch, features) = input.dim();
        let input_3d = input
            .clone()
            .into_shape((batch, features, 1))
            .map_err(|e| {
                KwaversError::System(crate::error::SystemError::MemoryAllocation {
                    requested_bytes: batch * features * std::mem::size_of::<f32>(),
                    reason: e.to_string(),
                })
            })?;

        // Run inference
        let output_3d = self.engine.infer_batch(&input_3d)?;

        // Convert back to 2D
        let output_2d = output_3d.index_axis(Axis(2), 0).to_owned();
        Ok(output_2d)
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn update(&mut self, gradients: &Array2<f32>) -> KwaversResult<()> {
        // Gradient descent update
        let learning_rate = 1e-3_f32;

        if gradients.shape() != self.engine.weights_mut().shape() {
            return Err(KwaversError::Physics(
                crate::error::PhysicsError::DimensionMismatch,
            ));
        }

        let weights = self.engine.weights_mut();
        *weights -= &(gradients * learning_rate);
        Ok(())
    }
}

/// Anomaly detection model
#[derive(Debug)]
pub struct AnomalyDetectorModel {
    pub(crate) engine: InferenceEngine,
    metadata: ModelMetadata,
}

impl AnomalyDetectorModel {
    /// Create a randomly initialized anomaly detector
    pub fn with_random_weights(features: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = Array2::from_shape_fn((features, 1), |_| rng.gen_range(-0.05..0.05));
        let engine = InferenceEngine::from_weights(weights, None, 32, false);

        let metadata = ModelMetadata {
            name: "AnomalyDetector".to_string(),
            version: "1.0.0".to_string(),
            input_shape: vec![features],
            output_shape: vec![1],
            accuracy: 0.0,
            inference_time_ms: 0.0,
        };

        Self { engine, metadata }
    }

    /// Create from explicit weights and bias
    pub fn from_weights(weights: Array2<f32>, bias: Option<Array1<f32>>) -> Self {
        let (features, _) = weights.dim();
        let engine = InferenceEngine::from_weights(weights, bias, 32, false);

        let metadata = ModelMetadata {
            name: "AnomalyDetector".to_string(),
            version: "1.0.0".to_string(),
            input_shape: vec![features],
            output_shape: vec![1],
            accuracy: 0.0,
            inference_time_ms: 0.0,
        };

        Self { engine, metadata }
    }
}

impl MLModel for AnomalyDetectorModel {
    fn model_type(&self) -> ModelType {
        ModelType::AnomalyDetector
    }

    fn infer(&self, input: &Array2<f32>) -> KwaversResult<Array2<f32>> {
        // Convert 2D input to 3D for the engine
        let (batch, features) = input.dim();
        let input_3d = input
            .clone()
            .into_shape((batch, features, 1))
            .map_err(|e| {
                KwaversError::System(crate::error::SystemError::MemoryAllocation {
                    requested_bytes: batch * features * std::mem::size_of::<f32>(),
                    reason: e.to_string(),
                })
            })?;

        // Run inference
        let output_3d = self.engine.infer_batch(&input_3d)?;

        // Convert back to 2D
        let output_2d = output_3d.index_axis(Axis(2), 0).to_owned();
        Ok(output_2d)
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn update(&mut self, gradients: &Array2<f32>) -> KwaversResult<()> {
        // Gradient descent update
        let learning_rate = 1e-3_f32;

        if gradients.shape() != self.engine.weights_mut().shape() {
            return Err(KwaversError::Physics(
                crate::error::PhysicsError::DimensionMismatch,
            ));
        }

        let weights = self.engine.weights_mut();
        *weights -= &(gradients * learning_rate);
        Ok(())
    }
}

/// Convergence prediction model
#[derive(Debug)]
pub struct ConvergencePredictorModel {
    pub(crate) engine: InferenceEngine,
    metadata: ModelMetadata,
}

impl ConvergencePredictorModel {
    /// Create a randomly initialized convergence predictor
    pub fn with_random_weights(features: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = Array2::from_shape_fn((features, 1), |_| rng.gen_range(-0.1..0.1));
        let engine = InferenceEngine::from_weights(weights, None, 64, false);

        let metadata = ModelMetadata {
            name: "ConvergencePredictor".to_string(),
            version: "1.0.0".to_string(),
            input_shape: vec![features],
            output_shape: vec![1],
            accuracy: 0.0,
            inference_time_ms: 0.0,
        };

        Self { engine, metadata }
    }

    /// Create from explicit weights and bias
    pub fn from_weights(weights: Array2<f32>, bias: Option<Array1<f32>>) -> Self {
        let (features, _) = weights.dim();
        let engine = InferenceEngine::from_weights(weights, bias, 64, false);

        let metadata = ModelMetadata {
            name: "ConvergencePredictor".to_string(),
            version: "1.0.0".to_string(),
            input_shape: vec![features],
            output_shape: vec![1],
            accuracy: 0.0,
            inference_time_ms: 0.0,
        };

        Self { engine, metadata }
    }
}

impl MLModel for ConvergencePredictorModel {
    fn model_type(&self) -> ModelType {
        ModelType::ConvergencePredictor
    }

    fn infer(&self, input: &Array2<f32>) -> KwaversResult<Array2<f32>> {
        // Convert 2D input to 3D for the engine
        let (batch, features) = input.dim();
        let input_3d = input
            .clone()
            .into_shape((batch, features, 1))
            .map_err(|e| {
                KwaversError::System(crate::error::SystemError::MemoryAllocation {
                    requested_bytes: batch * features * std::mem::size_of::<f32>(),
                    reason: e.to_string(),
                })
            })?;

        // Run inference
        let output_3d = self.engine.infer_batch(&input_3d)?;

        // Convert back to 2D
        let output_2d = output_3d.index_axis(Axis(2), 0).to_owned();
        Ok(output_2d)
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn update(&mut self, gradients: &Array2<f32>) -> KwaversResult<()> {
        // Gradient descent update
        let learning_rate = 1e-3_f32;

        if gradients.shape() != self.engine.weights_mut().shape() {
            return Err(KwaversError::Physics(
                crate::error::PhysicsError::DimensionMismatch,
            ));
        }

        let weights = self.engine.weights_mut();
        *weights -= &(gradients * learning_rate);
        Ok(())
    }
}

/// Outcome predictor model (binary logistic regression)
pub struct OutcomePredictorModel {
    weights: Array1<f32>,
    bias: f32,
    metadata: ModelMetadata,
}

impl OutcomePredictorModel {
    /// Create predictor with explicit weights and bias (features length must match weights length).
    pub fn from_weights(weights: Array1<f32>, bias: f32) -> Self {
        let features = weights.len();
        let metadata = ModelMetadata {
            name: "OutcomePredictor".to_string(),
            version: "1.0.0".to_string(),
            input_shape: vec![features],
            output_shape: vec![2], // binary outcome {0,1}
            accuracy: 0.0,
            inference_time_ms: 0.0,
        };
        Self {
            weights,
            bias,
            metadata,
        }
    }

    /// Sigmoid function
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
}

impl MLModel for OutcomePredictorModel {
    fn model_type(&self) -> ModelType {
        ModelType::ConvergencePredictor
    }

    fn infer(&self, input: &Array2<f32>) -> KwaversResult<Array2<f32>> {
        // Input validation
        let (samples, features) = input.dim();
        if features != self.metadata.input_shape[0] {
            return Err(KwaversError::Physics(
                crate::error::PhysicsError::DimensionMismatch,
            ));
        }

        // Compute raw logit = x·w + b
        let logits: Array1<f32> = input
            .dot(&self.weights.view().insert_axis(Axis(1)))
            .index_axis(Axis(1), 0)
            .to_owned()
            + self.bias;

        // Convert to probabilities using sigmoid; output two-class probabilities [p0, p1]
        let mut probs = Array2::<f32>::zeros((samples, 2));
        for (i, &logit) in logits.iter().enumerate() {
            let p1 = Self::sigmoid(logit);
            let p0 = 1.0 - p1;
            probs[(i, 0)] = p0;
            probs[(i, 1)] = p1;
        }
        Ok(probs)
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn update(&mut self, _gradients: &Array2<f32>) -> KwaversResult<()> {
        // Online learning not yet implemented for predictor
        Err(KwaversError::NotImplemented(
            "Online update for OutcomePredictor".to_string(),
        ))
    }
}
