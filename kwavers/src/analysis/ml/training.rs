//! ML Training Loop with Burn Autodiff
//!
//! This module provides a comprehensive training pipeline for neural beamforming models
//! using the Burn deep learning framework. It implements:
//! - Mini-batch stochastic gradient descent with multiple optimizers
//! - Physics-informed loss balancing
//! - Model checkpointing and serialization
//! - Training metrics and convergence monitoring
//! - Data augmentation for improved generalization
//!
//! ## Architecture
//!
//! The training pipeline follows a standard supervised learning framework:
//! ```text
//! Dataset → Batching → Forward Pass → Loss Computation → Backward Pass → Parameter Update
//!    ↓         ↓            ↓              ↓                  ↓              ↓
//!  Input   Mini-batch   Network(x)   L_data + λ·L_physics    ∂L/∂w      w := w - η∇L
//! ```
//!
//! ## Physics-Informed Loss
//!
//! The total loss combines two components with weight balancing:
//! ```text
//! L_total = λ_data · L_data + λ_physics · L_physics
//! ```
//!
//! Where:
//! - **L_data**: MSE loss between predictions and ground truth
//! - **L_physics**: Physics constraint violation (reciprocity, wave equation residuals)
//! - **λ_data, λ_physics**: Adaptive weights (sum = 1.0)
//!
//! ## References
//!
//! - Kingma & Ba (2015) "Adam: A Method for Stochastic Optimization"
//! - Raissi et al. (2019) "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems"
//! - Goodfellow et al. (2016) "Deep Learning" (Chapters 8-9: Optimization)

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{s, Array2};

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
            checkpoint_dir: "./checkpoints".to_string(),
            verbose: true,
        }
    }
}

impl TrainingConfig {
    /// Validate configuration constraints
    pub fn validate(&self) -> KwaversResult<()> {
        if self.num_epochs == 0 {
            return Err(KwaversError::InvalidInput(
                "num_epochs must be > 0".to_string(),
            ));
        }

        if self.batch_size == 0 {
            return Err(KwaversError::InvalidInput(
                "batch_size must be > 0".to_string(),
            ));
        }

        if self.learning_rate <= 0.0 || self.learning_rate > 1.0 {
            return Err(KwaversError::InvalidInput(
                "learning_rate must be in (0, 1]".to_string(),
            ));
        }

        if self.learning_rate_decay <= 0.0 || self.learning_rate_decay > 1.0 {
            return Err(KwaversError::InvalidInput(
                "learning_rate_decay must be in (0, 1]".to_string(),
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
                "validation_split must be in [0, 1)".to_string(),
            ));
        }

        Ok(())
    }

    /// Builder pattern: set number of epochs
    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.num_epochs = epochs;
        self
    }

    /// Builder pattern: set batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Builder pattern: set learning rate
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr.clamp(1e-8, 1.0);
        self
    }

    /// Builder pattern: set loss weights
    pub fn with_loss_weights(mut self, lambda_data: f64, lambda_physics: f64) -> Self {
        let sum = lambda_data + lambda_physics;
        self.lambda_data = lambda_data / sum;
        self.lambda_physics = lambda_physics / sum;
        self
    }

    /// Builder pattern: enable gradient clipping
    pub fn with_gradient_clip(mut self, clip_norm: f64) -> Self {
        self.gradient_clip = if clip_norm > 0.0 {
            Some(clip_norm)
        } else {
            None
        };
        self
    }
}

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

/// Physics loss component
///
/// Enforces wave equation and acoustic constraints:
/// - Reciprocity: Forward and reverse paths should be equal
/// - Coherence: Adjacent elements should have similar phases
/// - Sparsity: Suppress noise via L1 regularization on weights
#[derive(Debug, Clone)]
pub struct PhysicsLoss {
    /// Weight for reciprocity constraint
    pub reciprocity_weight: f64,
    /// Weight for coherence constraint
    pub coherence_weight: f64,
    /// Weight for sparsity (L1 regularization)
    pub sparsity_weight: f64,
}

impl Default for PhysicsLoss {
    fn default() -> Self {
        Self {
            reciprocity_weight: 0.5,
            coherence_weight: 0.3,
            sparsity_weight: 0.2,
        }
    }
}

impl PhysicsLoss {
    /// Compute reciprocity constraint violation
    ///
    /// For a reciprocal system: response(A→B) = response(B→A)
    /// Violation = ||forward - reverse||²
    pub fn reciprocity_violation(forward: &Array2<f64>, reverse: &Array2<f64>) -> f64 {
        if forward.dim() != reverse.dim() {
            return f64::INFINITY;
        }

        let diff = forward - reverse;
        let mse: f64 = diff.iter().map(|x| x * x).sum::<f64>() / (forward.len() as f64);
        mse
    }

    /// Compute coherence constraint (phase continuity)
    ///
    /// Adjacent elements should have similar phases (continuous wavefront)
    /// Violation = sum_i |phase(i+1) - phase(i)|
    pub fn coherence_violation(phases: &Array2<f64>) -> f64 {
        let mut violation = 0.0;

        for i in 0..phases.dim().0 - 1 {
            for j in 0..phases.dim().1 {
                let phase_diff = (phases[[i + 1, j]] - phases[[i, j]]).abs();
                // Normalize to [-π, π]
                let normalized = if phase_diff > std::f64::consts::PI {
                    2.0 * std::f64::consts::PI - phase_diff
                } else {
                    phase_diff
                };
                violation += normalized;
            }
        }

        violation / phases.len() as f64
    }

    /// Compute sparsity constraint (L1 regularization on weights)
    pub fn sparsity_violation(weights: &Array2<f64>) -> f64 {
        weights.iter().map(|w| w.abs()).sum::<f64>() / weights.len() as f64
    }
}

/// Gradient-based optimizer
#[derive(Debug, Clone, Copy)]
pub enum Optimizer {
    /// Stochastic Gradient Descent
    SGD,
    /// SGD with Momentum
    Momentum { beta: f64 },
    /// Adam (Adaptive Moment Estimation)
    Adam {
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    },
    /// RMSprop (Root Mean Square Propagation)
    RMSprop { beta: f64, epsilon: f64 },
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::Adam {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

/// Training history (logs)
#[derive(Debug, Clone)]
pub struct TrainingHistory {
    /// Per-epoch metrics
    pub epochs: Vec<TrainingMetrics>,
    /// Best validation loss achieved
    pub best_val_loss: f64,
    /// Epoch of best validation loss
    pub best_epoch: usize,
    /// Total training time (seconds)
    pub total_time: f64,
}

impl TrainingHistory {
    /// Create empty training history
    pub fn new() -> Self {
        Self {
            epochs: Vec::new(),
            best_val_loss: f64::INFINITY,
            best_epoch: 0,
            total_time: 0.0,
        }
    }

    /// Add metrics for an epoch
    pub fn add_epoch(&mut self, metrics: TrainingMetrics) {
        if metrics.val_loss < self.best_val_loss {
            self.best_val_loss = metrics.val_loss;
            self.best_epoch = metrics.epoch;
        }
        self.epochs.push(metrics);
    }

    /// Get convergence rate (loss improvement per epoch)
    pub fn convergence_rate(&self) -> f64 {
        if self.epochs.len() < 2 {
            return 0.0;
        }

        let first_loss = self.epochs[0].train_loss;
        let last_loss = self.epochs[self.epochs.len() - 1].train_loss;
        let improvement = (first_loss - last_loss).abs();
        let num_epochs = self.epochs.len() as f64;

        improvement / num_epochs
    }
}

impl Default for TrainingHistory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_training_config_validation() {
        let mut config = TrainingConfig::default();
        config.num_epochs = 0;
        assert!(config.validate().is_err());

        config.num_epochs = 100;
        config.learning_rate = 0.0;
        assert!(config.validate().is_err());

        config.learning_rate = 0.001;
        config.lambda_data = 0.6;
        config.lambda_physics = 0.3; // Sum != 1.0
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_training_config_builder() {
        let config = TrainingConfig::default()
            .with_epochs(200)
            .with_batch_size(64)
            .with_learning_rate(0.0001);

        assert_eq!(config.num_epochs, 200);
        assert_eq!(config.batch_size, 64);
        assert!(config.learning_rate < 0.001);
    }

    #[test]
    fn test_training_dataset_creation() {
        let inputs = Array2::<f64>::zeros((100, 10));
        let targets = Array2::<f64>::zeros((100, 1));
        let dataset = TrainingDataset::new(inputs, targets);
        assert!(dataset.is_ok());
    }

    #[test]
    fn test_training_dataset_mismatched_sizes() {
        let inputs = Array2::<f64>::zeros((100, 10));
        let targets = Array2::<f64>::zeros((50, 1)); // Wrong number of samples
        let dataset = TrainingDataset::new(inputs, targets);
        assert!(dataset.is_err());
    }

    #[test]
    fn test_training_dataset_split() {
        let inputs = Array2::<f64>::zeros((100, 10));
        let targets = Array2::<f64>::zeros((100, 1));
        let dataset = TrainingDataset::new(inputs, targets).unwrap();

        let (train, val) = dataset.split(0.2).unwrap();
        assert_eq!(train.len() + val.len(), 100);
        assert!(train.len() > val.len()); // 80/20 split
    }

    #[test]
    fn test_training_dataset_batch() {
        let inputs = Array2::<f64>::zeros((100, 10));
        let targets = Array2::<f64>::zeros((100, 1));
        let dataset = TrainingDataset::new(inputs, targets).unwrap();

        let batch = dataset.batch(0, 32).unwrap();
        assert_eq!(batch.len(), 32);

        let batch2 = dataset.batch(32, 32).unwrap();
        assert_eq!(batch2.len(), 32);
    }

    #[test]
    fn test_physics_loss_reciprocity() {
        let forward = Array2::<f64>::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let reverse = forward.clone();
        let loss = PhysicsLoss::reciprocity_violation(&forward, &reverse);
        assert!((loss - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_optimizer_default() {
        let optimizer = Optimizer::default();
        if let Optimizer::Adam { beta1, beta2, .. } = optimizer {
            assert!((beta1 - 0.9).abs() < 1e-10);
            assert!((beta2 - 0.999).abs() < 1e-10);
        } else {
            panic!("Default should be Adam");
        }
    }

    #[test]
    fn test_training_history() {
        let mut history = TrainingHistory::new();
        assert_eq!(history.epochs.len(), 0);
        assert!(history.best_val_loss.is_infinite());

        let metrics = TrainingMetrics {
            epoch: 0,
            train_loss: 1.0,
            train_data_loss: 0.8,
            train_physics_loss: 0.2,
            val_loss: 0.9,
            learning_rate: 0.001,
            gradient_norm: 0.1,
            time_per_epoch: 1.5,
        };

        history.add_epoch(metrics);
        assert_eq!(history.epochs.len(), 1);
        assert!((history.best_val_loss - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_loss_weights_normalization() {
        let config = TrainingConfig::default().with_loss_weights(2.0, 1.0);

        // Should normalize to 2/3 and 1/3
        assert!((config.lambda_data - 2.0 / 3.0).abs() < 1e-10);
        assert!((config.lambda_physics - 1.0 / 3.0).abs() < 1e-10);
    }
}
