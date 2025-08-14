//! Model training pipeline with data augmentation

use crate::error::{KwaversError, KwaversResult};
use crate::ml::MLModel;
use ndarray::{Array1, Array2, Array3, Axis};
use rand::seq::SliceRandom;
use rand::thread_rng;
use crate::ml::models::TissueClassifierModel;

/// Training pipeline for ML models
pub struct TrainingPipeline {
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
}

impl TrainingPipeline {
    pub fn new(epochs: usize, batch_size: usize, learning_rate: f64) -> Self {
        Self {
            epochs,
            batch_size,
            learning_rate,
        }
    }
    
    /// Train the provided TissueClassifierModel on `(samples, features, 1)` data
    /// with integer class labels stored as `(samples, 1, 1)`.  Returns the loss
    /// at each epoch.
    pub fn train(
        &self,
        model: &mut TissueClassifierModel,
        training_data: &Array3<f64>,
        labels: &Array3<u8>,
    ) -> KwaversResult<Vec<f32>> {
        // Input validation
        let (samples, features, depth) = training_data.dim();
        if depth != 1 {
            return Err(KwaversError::Physics(crate::error::PhysicsError::DimensionMismatch));
        }

        let (label_samples, _, _) = labels.dim();
        if label_samples != samples {
            return Err(KwaversError::Validation(
                crate::error::ValidationError::RangeValidation {
                    field: "labels.samples".to_string(),
                    value: label_samples.to_string(),
                    min: samples.to_string(),
                    max: samples.to_string(),
                },
            ));
        }

        // Convert to f32 and reshape to 2-D matrices
        let inputs_f32 = training_data.mapv(|v| v as f32);
        let inputs_2d = inputs_f32.index_axis(Axis(2), 0).to_owned(); // (samples, features)

        let targets: Array1<usize> = labels
            .iter()
            .map(|&v| v as usize)
            .collect::<Vec<_>>()
            .into();

        let classes = *targets.iter().max().unwrap_or(&0) + 1;

        // If model output dimension differs, recreate weights
        {
            let metadata = model.metadata();
            if metadata.output_shape[0] != classes {
                // Reinitialise model with matching class count
                let weights = Array2::<f32>::zeros((features, classes));
                *model = TissueClassifierModel::from_weights(weights, None);
            }
        }

        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..samples).collect();
        let mut losses = Vec::with_capacity(self.epochs);

        for _epoch in 0..self.epochs {
            // Shuffle each epoch
            indices.shuffle(&mut rng);

            let mut epoch_loss = 0f32;
            for batch_start in (0..samples).step_by(self.batch_size) {
                let batch_end = (batch_start + self.batch_size).min(samples);
                let batch_idx = &indices[batch_start..batch_end];

                // Gather batch
                let batch_inputs = inputs_2d.select(Axis(0), batch_idx);
                let batch_targets: Array1<usize> = batch_idx
                    .iter()
                    .map(|&i| targets[i])
                    .collect::<Vec<_>>()
                    .into();

                // Forward pass
                let logits = batch_inputs.dot(model.engine.weights_mut());

                // Softmax
                let mut probs = logits.map(|&v| v.exp());
                for mut row in probs.rows_mut() {
                    let sum = row.sum();
                    row.mapv_inplace(|v| v / sum);
                }

                // Loss & Gradient
                let mut grad = probs.clone();
                for (mut row, &tgt) in grad.rows_mut().into_iter().zip(batch_targets.iter()) {
                    row[tgt] -= 1.0;
                }

                let batch_size_f32 = batch_targets.len() as f32;
                grad.mapv_inplace(|v| v / batch_size_f32);

                // Backprop: dL/dW = X^T · grad
                let grad_w = batch_inputs.t().dot(&grad);

                // SGD update
                let weights = model.engine.weights_mut();
                *weights -= &(grad_w * self.learning_rate as f32);

                // Cross entropy loss
                for (i, &tgt) in batch_targets.iter().enumerate() {
                    let p = probs[(i, tgt)].max(1e-8);
                    epoch_loss += -p.ln();
                }
            }

            losses.push(epoch_loss / samples as f32);
        }

        Ok(losses)
    }
}