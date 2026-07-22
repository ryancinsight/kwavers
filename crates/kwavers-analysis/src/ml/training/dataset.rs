//! Training dataset and batch operations.

use kwavers_core::error::{KwaversError, KwaversResult};
use leto::{Array2, SliceArg};

/// Training metrics for monitoring convergence
#[derive(Debug, Clone)]
pub struct EpochTrainingMetrics {
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
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn new(inputs: Array2<f64>, targets: Array2<f64>) -> KwaversResult<Self> {
        if inputs.shape()[0] != targets.shape()[0] {
            return Err(KwaversError::InvalidInput(
                "inputs and targets must have same number of samples".to_owned(),
            ));
        }
        Ok(Self { inputs, targets })
    }

    /// Get number of samples
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn len(&self) -> usize {
        self.inputs.shape()[0]
    }

    /// Check if dataset is empty
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Split dataset into training and validation sets
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn split(&self, validation_fraction: f64) -> KwaversResult<(Self, Self)> {
        if validation_fraction <= 0.0 || validation_fraction >= 1.0 {
            return Err(KwaversError::InvalidInput(
                "validation_fraction must be in (0, 1)".to_owned(),
            ));
        }

        let n = self.len();
        let val_size = ((n as f64) * validation_fraction).ceil() as usize;
        let train_size = n - val_size;

        let head = [
            SliceArg::Range {
                start: Some(0),
                end: Some(train_size as isize),
                step: 1,
            },
            SliceArg::All,
        ];
        let tail = [
            SliceArg::Range {
                start: Some(train_size as isize),
                end: None,
                step: 1,
            },
            SliceArg::All,
        ];
        let train_inputs = self
            .inputs
            .slice_with::<2>(&head)
            .expect("invariant: train row range in bounds")
            .to_contiguous();
        let train_targets = self
            .targets
            .slice_with::<2>(&head)
            .expect("invariant: train row range in bounds")
            .to_contiguous();
        let val_inputs = self
            .inputs
            .slice_with::<2>(&tail)
            .expect("invariant: validation row range in bounds")
            .to_contiguous();
        let val_targets = self
            .targets
            .slice_with::<2>(&tail)
            .expect("invariant: validation row range in bounds")
            .to_contiguous();

        Ok((
            Self {
                inputs: train_inputs,
                targets: train_targets,
            },
            Self {
                inputs: val_inputs,
                targets: val_targets,
            },
        ))
    }

    /// Get batch of specified size starting at offset
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn batch(&self, offset: usize, batch_size: usize) -> KwaversResult<Self> {
        let n = self.len();
        if offset >= n {
            return Err(KwaversError::InvalidInput(
                "offset exceeds dataset size".to_owned(),
            ));
        }

        let end = (offset + batch_size).min(n);
        let rows = [
            SliceArg::Range {
                start: Some(offset as isize),
                end: Some(end as isize),
                step: 1,
            },
            SliceArg::All,
        ];
        let batch_inputs = self
            .inputs
            .slice_with::<2>(&rows)
            .expect("invariant: batch row range in bounds")
            .to_contiguous();
        let batch_targets = self
            .targets
            .slice_with::<2>(&rows)
            .expect("invariant: batch row range in bounds")
            .to_contiguous();

        Ok(Self {
            inputs: batch_inputs,
            targets: batch_targets,
        })
    }
}
