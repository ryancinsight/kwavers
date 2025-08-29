//! Tissue classification model

use super::{MLModel, ModelMetadata};
use crate::error::KwaversResult;
use crate::ml::inference::InferenceEngine;
use ndarray::{Array1, Array2};

/// Tissue classification model for acoustic simulations
#[derive(Debug)]
pub struct TissueClassifierModel {
    pub(crate) engine: InferenceEngine,
    metadata: ModelMetadata,
}

impl TissueClassifierModel {
    /// Load model from path
    pub fn load(path: &std::path::Path) -> KwaversResult<Self> {
        // Simplified loading - real implementation would deserialize weights
        Ok(Self::with_random_weights(128, 10))
    }
    
    /// Create from weights
    pub fn from_weights(weights: Array2<f32>, bias: Option<Array1<f32>>) -> Self {
        let (features, classes) = weights.dim();
        let engine = InferenceEngine::from_weights(weights, bias, 32, false);
        
        let metadata = ModelMetadata {
            name: "TissueClassifier".to_string(),
            version: "1.0.0".to_string(),
            input_shape: vec![features],
            output_shape: vec![classes],
            accuracy: 0.0_f64,
            inference_time_ms: 0.0_f64,
        };
        
        Self { engine, metadata }
    }
    
    /// Get metadata
    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }
    
    /// Run inference 
    pub fn infer(&self, input: &Array2<f32>) -> KwaversResult<Array2<f32>> {
        self.predict(input)
    }
    
    /// Create a classifier with random weights
    pub fn with_random_weights(features: usize, classes: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let weights = Array2::from_shape_fn((features, classes), |_| rng.gen_range(-0.05..0.05));
        let engine = InferenceEngine::from_weights(weights, None, 32, false);

        let metadata = ModelMetadata {
            name: "TissueClassifier".to_string(),
            version: "1.0.0".to_string(),
            input_shape: vec![features],
            output_shape: vec![classes],
            accuracy: 0.0_f64,
            inference_time_ms: 0.0_f64,
        };

        Self { engine, metadata }
    }

    /// Classify tissue types
    pub fn classify(&self, features: &Array1<f32>) -> KwaversResult<usize> {
        let input = features.clone().insert_axis(ndarray::Axis(0));
        let output = self.engine.forward(&input)?;

        // Find class with highest probability
        let class = output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        Ok(class)
    }
}

impl MLModel for TissueClassifierModel {
    fn predict(&self, input: &Array2<f32>) -> KwaversResult<Array2<f32>> {
        self.engine.forward(input)
    }

    fn accuracy(&self) -> f32 {
        self.metadata.accuracy
    }

    fn name(&self) -> &str {
        &self.metadata.name
    }
}
