//! Pre-trained models for tissue classification and parameter optimization

use crate::error::{KwaversError, KwaversResult};
use crate::ml::{inference::InferenceEngine, MLModel, ModelMetadata, ModelType};
use ndarray::{Array1, Array2, Axis};
use rand::Rng;

/// Tissue classification model
pub struct TissueClassifierModel {
    engine: InferenceEngine,
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
        let engine = InferenceEngine::from_weights(weights, bias.clone(), 32, false);

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
            return Err(KwaversError::Validation(
                crate::error::ValidationError::DimensionMismatch,
            ));
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
            return Err(KwaversError::Validation(
                crate::error::ValidationError::DimensionMismatch,
            ));
        }
        let weights = self.engine.weights_mut();
        *weights -= &(gradients * learning_rate);
        Ok(())
    }
}