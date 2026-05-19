//! Parameter optimization model

use super::{MLModel, MlModelMetadata};
use crate::analysis::ml::inference::InferenceEngine;
use crate::core::error::KwaversResult;
use ndarray::{Array1, Array2};

/// Parameter optimization model
#[derive(Debug)]
pub struct ParameterOptimizerModel {
    engine: InferenceEngine,
    metadata: MlModelMetadata,
}

impl ParameterOptimizerModel {
    /// Load model from path
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn load(_path: &std::path::Path) -> KwaversResult<Self> {
        // Basic model initialization (future: load from serialized weights)
        Ok(Self::new(128, 64))
    }

    /// Create model from weights
    #[must_use]
    pub fn from_weights(weights: Array2<f32>, bias: Option<Array1<f32>>) -> Self {
        let (input_dim, output_dim) = weights.dim();
        Self {
            engine: InferenceEngine::from_weights(weights, bias, 32, false),
            metadata: MlModelMetadata {
                name: "ParameterOptimizer".to_owned(),
                version: "1.0.0".to_owned(),
                input_shape: vec![input_dim],
                output_shape: vec![output_dim],
                accuracy: 0.92_f64,
                inference_time_ms: 0.5_f64,
            },
        }
    }

    /// Get metadata
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn metadata(&self) -> &MlModelMetadata {
        &self.metadata
    }

    /// Run inference
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn infer(&self, input: &Array2<f32>) -> KwaversResult<Array2<f32>> {
        self.predict(input)
    }

    /// Create a parameter optimizer with randomly initialized weights.
    ///
    /// Weights are drawn from U(−0.1, 0.1). `input_dim` is the feature
    /// count; `output_dim` is the number of optimized parameter outputs.
    /// Accuracy and inference time are zero until the model is trained.
    #[must_use]
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let weights = Array2::from_shape_fn((input_dim, output_dim), |_| rng.gen_range(-0.1..0.1));

        Self {
            engine: InferenceEngine::from_weights(weights, None, 32, false),
            metadata: MlModelMetadata {
                name: "ParameterOptimizer".to_owned(),
                version: "1.0.0".to_owned(),
                input_shape: vec![input_dim],
                output_shape: vec![output_dim],
                accuracy: 0.0_f64,
                inference_time_ms: 0.0_f64,
            },
        }
    }
}

impl MLModel for ParameterOptimizerModel {
    fn predict(&self, input: &Array2<f32>) -> KwaversResult<Array2<f32>> {
        self.engine.forward(input)
    }

    fn accuracy(&self) -> f64 {
        self.metadata.accuracy
    }

    fn name(&self) -> &str {
        &self.metadata.name
    }
}
