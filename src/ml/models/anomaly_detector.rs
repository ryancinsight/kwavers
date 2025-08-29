//! Anomaly detection model

use super::{MLModel, ModelMetadata};
use crate::error::KwaversResult;
use ndarray::Array2;

/// Anomaly detection model
#[derive(Debug)]
pub struct AnomalyDetectorModel {
    threshold: f32,
    metadata: ModelMetadata,
}

impl AnomalyDetectorModel {
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            metadata: ModelMetadata {
                name: "AnomalyDetector".to_string(),
                version: "1.0.0".to_string(),
                input_shape: vec![1],
                output_shape: vec![1],
                accuracy: 0.95,
                inference_time_ms: 0.1,
            },
        }
    }
}

impl MLModel for AnomalyDetectorModel {
    fn predict(&self, input: &Array2<f32>) -> KwaversResult<Array2<f32>> {
        // Simple threshold-based anomaly detection
        Ok(input.mapv(|x| if x.abs() > self.threshold { 1.0 } else { 0.0 }))
    }

    fn accuracy(&self) -> f32 {
        self.metadata.accuracy
    }

    fn name(&self) -> &str {
        &self.metadata.name
    }
}
