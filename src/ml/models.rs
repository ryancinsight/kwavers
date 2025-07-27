//! Pre-trained models for tissue classification and parameter optimization

use crate::error::{KwaversError, KwaversResult};
use crate::ml::{inference::InferenceEngine, MLModel, ModelMetadata, ModelType};
use ndarray::{Array1, Array2, Axis};
use rand::Rng;

/// Tissue classification model
pub struct TissueClassifierModel {
    pub(crate) engine: InferenceEngine,
    metadata: ModelMetadata,
}

impl TissueClassifierModel {
    /// Create a *randomly initialised* classifier with the given dimensionality.
    pub fn new_random(features: usize, classes: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = Array2::from_shape_fn((features, classes), |_| rng.gen_range(-0.05..0.05));
        let engine = InferenceEngine::from_weights(weights, None, 32, false);

        let metadata = ModelMetadata {
            name: "RandomTissueClassifier".to_string(),
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
        let (batch, features) = input.dim();
        if features != self.metadata.input_shape[0] {
            return Err(KwaversError::Physics(crate::error::PhysicsError::DimensionMismatch));
        }

        // Reshape into the 3-D tensor expected by the engine: (batch, features, 1)
        let input3 = input.clone().into_shape((batch, features, 1)).map_err(|_| {
            KwaversError::Validation(crate::error::ValidationError::TypeValidation {
                field: "input".to_string(),
                expected_type: "(batch, features)".to_string(),
                actual_type: format!("({}, {})", batch, features),
            })
        })?;

        let output3 = self.engine.infer_batch(&input3)?;
        let output2 = output3.index_axis(Axis(2), 0).to_owned();
        Ok(output2)
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn update(&mut self, gradients: &Array2<f32>) -> KwaversResult<()> {
        // A minimal online-learning step: stochastic gradient descent with a
        // fixed learning rate.
        let learning_rate = 1e-3_f32;

        // Sanity-check dimensionality
        if gradients.shape() != self.engine.weights_mut().shape() {
            return Err(KwaversError::Physics(crate::error::PhysicsError::DimensionMismatch));
        }
        let weights = self.engine.weights_mut();
        *weights -= &(gradients * learning_rate);
        Ok(())
    }
}

/// Simple outcome predictor model (binary logistic regression)
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
        Self { weights, bias, metadata }
    }

    /// Sigmoid function
    fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
}

impl MLModel for OutcomePredictorModel {
    fn model_type(&self) -> ModelType {
        ModelType::ConvergencePredictor
    }

    fn infer(&self, input: &Array2<f32>) -> KwaversResult<Array2<f32>> {
        // Input validation
        let (samples, features) = input.dim();
        if features != self.metadata.input_shape[0] {
            return Err(KwaversError::Physics(crate::error::PhysicsError::DimensionMismatch));
        }

        // Compute raw logit = x·w + b
        let logits: Array1<f32> = input
            .dot(&self.weights.view().insert_axis(Axis(1)))
            .index_axis(Axis(1), 0)
            .to_owned()
            + self.bias.clone();

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

    fn metadata(&self) -> &ModelMetadata { &self.metadata }

    fn update(&mut self, _gradients: &Array2<f32>) -> KwaversResult<()> {
        // Online learning not yet implemented for predictor
        Err(KwaversError::NotImplemented("Online update for OutcomePredictor".to_string()))
    }
}