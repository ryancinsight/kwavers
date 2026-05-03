//! Training dataset and batch operations.

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{s, Array2};

/// Training metrics for monitoring convergence
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    /// Epoch number
    pub epoch: usize,
    /// Training loss (data + physics weighted)
    pub train_loss: f64,
    /// Training data loss component
    pub train_data_loss: f64,
    /// Training physics loss component
    pub train_physics_loss: f64,
    /// Validation loss
    pub val_loss: f64,
    /// Current learning rate
    pub learning_rate: f64,
    /// Gradient norm before clipping
    pub gradient_norm: f64,
    /// Time per epoch (seconds)
    pub time_per_epoch: f64,
}

/// Dataset for training
#[derive(Debug, Clone)]
pub struct TrainingDataset {
    /// Input features (N, num_features)
    pub inputs: Array2<f64>,
    /// Ground truth outputs (N, num_outputs)
    pub targets: Array2<f64>,
}

impl TrainingDataset {
    /// Create new training dataset
    pub fn new(inputs: Array2<f64>, targets: Array2<f64>) -> KwaversResult<Self> {
        if inputs.dim().0 != targets.dim().0 {
            return Err(KwaversError::InvalidInput(
                "inputs and targets must have same number of samples".to_string(),
            ));
        }
        Ok(Self { inputs, targets })
    }

    /// Get number of samples
    pub fn len(&self) -> usize {
        self.inputs.dim().0
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Split dataset into training and validation sets
    pub fn split(
        &self,
        validation_fraction: f64,
    ) -> KwaversResult<(TrainingDataset, TrainingDataset)> {
        if validation_fraction <= 0.0 || validation_fraction >= 1.0 {
            return Err(KwaversError::InvalidInput(
                "validation_fraction must be in (0, 1)".to_string(),
            ));
        }

        let n = self.len();
        let val_size = ((n as f64) * validation_fraction).ceil() as usize;
        let train_size = n - val_size;

        let train_inputs = self.inputs.slice(s![0..train_size, ..]).to_owned();
        let train_targets = self.targets.slice(s![0..train_size, ..]).to_owned();
        let val_inputs = self.inputs.slice(s![train_size.., ..]).to_owned();
        let val_targets = self.targets.slice(s![train_size.., ..]).to_owned();

        Ok((
            TrainingDataset {
                inputs: train_inputs,
                targets: train_targets,
            },
            TrainingDataset {
                inputs: val_inputs,
                targets: val_targets,
            },
        ))
    }

    /// Get batch of specified size starting at offset
    pub fn batch(&self, offset: usize, batch_size: usize) -> KwaversResult<TrainingDataset> {
        let n = self.len();
        if offset >= n {
            return Err(KwaversError::InvalidInput(
                "offset exceeds dataset size".to_string(),
            ));
        }

        let end = (offset + batch_size).min(n);
        let batch_inputs = self.inputs.slice(s![offset..end, ..]).to_owned();
        let batch_targets = self.targets.slice(s![offset..end, ..]).to_owned();

        Ok(TrainingDataset {
            inputs: batch_inputs,
            targets: batch_targets,
        })
    }
}
