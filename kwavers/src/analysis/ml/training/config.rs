//! Training configuration for ML models.

use crate::core::error::{KwaversError, KwaversResult};

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of training epochs
    pub num_epochs: usize,
    /// Batch size for mini-batch SGD
    pub batch_size: usize,
    /// Initial learning rate (step size for parameter updates)
    pub learning_rate: f64,
    /// Learning rate decay factor (applied after each epoch)
    pub learning_rate_decay: f64,
    /// Minimum learning rate (stop decay below this)
    pub min_learning_rate: f64,
    /// Weight for data loss term (relative to physics loss)
    pub lambda_data: f64,
    /// Weight for physics loss term (relative to data loss)
    pub lambda_physics: f64,
    /// Gradient clipping norm (prevent exploding gradients)
    pub gradient_clip: Option<f64>,
    /// Validation split fraction (0.0 - 1.0)
    pub validation_split: f64,
    /// Checkpoint frequency (save every N epochs)
    pub checkpoint_interval: usize,
    /// Checkpoint directory path
    pub checkpoint_dir: String,
    /// Enable verbose logging
    pub verbose: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            num_epochs: 100,
            batch_size: 32,
            learning_rate: 0.001,
            learning_rate_decay: 0.99,
            min_learning_rate: 1e-6,
            lambda_data: 0.8,
            lambda_physics: 0.2,
            gradient_clip: Some(10.0),
            validation_split: 0.2,
            checkpoint_interval: 10,
            checkpoint_dir: "./checkpoints".to_owned(),
            verbose: true,
        }
    }
}

impl TrainingConfig {
    /// Validate configuration constraints
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        if self.num_epochs == 0 {
            return Err(KwaversError::InvalidInput(
                "num_epochs must be > 0".to_owned(),
            ));
        }
        if self.batch_size == 0 {
            return Err(KwaversError::InvalidInput(
                "batch_size must be > 0".to_owned(),
            ));
        }
        if self.learning_rate <= 0.0 || self.learning_rate > 1.0 {
            return Err(KwaversError::InvalidInput(
                "learning_rate must be in (0, 1]".to_owned(),
            ));
        }
        if self.learning_rate_decay <= 0.0 || self.learning_rate_decay > 1.0 {
            return Err(KwaversError::InvalidInput(
                "learning_rate_decay must be in (0, 1]".to_owned(),
            ));
        }
        let lambda_sum = self.lambda_data + self.lambda_physics;
        if (lambda_sum - 1.0).abs() > 1e-6 {
            return Err(KwaversError::InvalidInput(format!(
                "lambda_data + lambda_physics must equal 1.0 (got {})",
                lambda_sum
            )));
        }
        if self.validation_split < 0.0 || self.validation_split >= 1.0 {
            return Err(KwaversError::InvalidInput(
                "validation_split must be in [0, 1)".to_owned(),
            ));
        }
        Ok(())
    }

    /// Builder pattern: set number of epochs
    #[must_use] 
    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.num_epochs = epochs;
        self
    }

    /// Builder pattern: set batch size
    #[must_use] 
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Builder pattern: set learning rate
    #[must_use] 
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr.clamp(1e-8, 1.0);
        self
    }

    /// Builder pattern: set loss weights
    #[must_use] 
    pub fn with_loss_weights(mut self, lambda_data: f64, lambda_physics: f64) -> Self {
        let sum = lambda_data + lambda_physics;
        self.lambda_data = lambda_data / sum;
        self.lambda_physics = lambda_physics / sum;
        self
    }

    /// Builder pattern: enable gradient clipping
    #[must_use] 
    pub fn with_gradient_clip(mut self, clip_norm: f64) -> Self {
        self.gradient_clip = if clip_norm > 0.0 {
            Some(clip_norm)
        } else {
            None
        };
        self
    }
}
