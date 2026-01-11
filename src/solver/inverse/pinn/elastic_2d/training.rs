//! Training loop and optimization for 2D Elastic Wave PINN
//!
//! This module implements the training procedure for physics-informed neural networks
//! solving the 2D elastic wave equation.
//!
//! # Status: Placeholder Implementation
//!
//! This is a minimal placeholder to allow compilation. Full Burn optimizer integration
//! is deferred pending resolution of API compatibility issues.
//!
//! # TODO (Task 2 Continuation)
//!
//! - Implement proper Burn 0.19+ optimizer API integration
//! - Add backward pass and gradient computation
//! - Implement learning rate scheduling
//! - Add training metrics and convergence monitoring
//! - Implement model checkpointing
//!
//! # Mathematical Foundation
//!
//! The training minimizes the physics-informed loss:
//!
//! ```text
//! θ* = arg min_θ L(θ) = arg min_θ [w_pde·L_pde + w_bc·L_bc + w_ic·L_ic + w_data·L_data]
//! ```
//!
//! where:
//! - L_pde: PDE residual loss (physics constraint)
//! - L_bc: Boundary condition loss
//! - L_ic: Initial condition loss
//! - L_data: Data fitting loss (for inverse problems)
//! - w_*: Loss weights
//!
//! The optimizer updates parameters using gradient descent:
//!
//! ```text
//! θ_{k+1} = θ_k - α_k ∇_θ L(θ_k)
//! ```
//!
//! where α_k is the learning rate at iteration k.

#[cfg(feature = "pinn")]
use super::config::{Config, LossWeights};

#[cfg(feature = "pinn")]
use super::loss::{BoundaryData, CollocationData, InitialData, LossComputer, ObservationData};

#[cfg(feature = "pinn")]
use super::model::ElasticPINN2D;

#[cfg(feature = "pinn")]
use crate::error::{KwaversError, KwaversResult};

#[cfg(feature = "pinn")]
use std::time::Instant;

#[cfg(feature = "pinn")]
use burn::{
    optim::{Adam, AdamConfig, Sgd, SgdConfig},
    tensor::backend::AutodiffBackend,
};

/// Training data container
///
/// Aggregates all data required for PINN training.
#[cfg(feature = "pinn")]
#[derive(Clone)]
pub struct TrainingData<B: AutodiffBackend> {
    /// Interior collocation points for PDE residual
    pub collocation: CollocationData<B>,
    /// Boundary condition data
    pub boundary: BoundaryData<B>,
    /// Initial condition data
    pub initial: InitialData<B>,
    /// Optional observation data (for inverse problems)
    pub observations: Option<ObservationData<B>>,
}

/// Training metrics tracked during optimization
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    /// Total loss history (one value per epoch)
    pub total_loss: Vec<f64>,
    /// PDE residual loss history
    pub pde_loss: Vec<f64>,
    /// Boundary condition loss history
    pub boundary_loss: Vec<f64>,
    /// Initial condition loss history
    pub initial_loss: Vec<f64>,
    /// Data fitting loss history
    pub data_loss: Vec<f64>,
    /// Training time per epoch (seconds)
    pub epoch_times: Vec<f64>,
    /// Total training time (seconds)
    pub total_time: f64,
    /// Number of epochs completed
    pub epochs_completed: usize,
    /// Learning rate history
    pub learning_rates: Vec<f64>,
}

impl TrainingMetrics {
    /// Create new empty metrics
    pub fn new() -> Self {
        Self {
            total_loss: Vec::new(),
            pde_loss: Vec::new(),
            boundary_loss: Vec::new(),
            initial_loss: Vec::new(),
            data_loss: Vec::new(),
            epoch_times: Vec::new(),
            total_time: 0.0,
            epochs_completed: 0,
            learning_rates: Vec::new(),
        }
    }

    /// Record metrics for current epoch
    pub fn record_epoch(
        &mut self,
        total: f64,
        pde: f64,
        boundary: f64,
        initial: f64,
        data: f64,
        lr: f64,
        epoch_time: f64,
    ) {
        self.total_loss.push(total);
        self.pde_loss.push(pde);
        self.boundary_loss.push(boundary);
        self.initial_loss.push(initial);
        self.data_loss.push(data);
        self.learning_rates.push(lr);
        self.epoch_times.push(epoch_time);
        self.epochs_completed += 1;
    }

    /// Get final loss value
    pub fn final_loss(&self) -> Option<f64> {
        self.total_loss.last().copied()
    }

    /// Get average epoch time
    pub fn average_epoch_time(&self) -> f64 {
        if self.epoch_times.is_empty() {
            0.0
        } else {
            self.epoch_times.iter().sum::<f64>() / self.epoch_times.len() as f64
        }
    }

    /// Check if training has converged
    ///
    /// Convergence criteria:
    /// - Loss change < tolerance for N consecutive epochs
    /// - Absolute loss < absolute tolerance
    pub fn has_converged(&self, tolerance: f64, window: usize) -> bool {
        if self.total_loss.len() < window {
            return false;
        }

        let recent = &self.total_loss[self.total_loss.len() - window..];
        let max = recent.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = recent.iter().cloned().fold(f64::INFINITY, f64::min);

        (max - min) < tolerance
    }
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// PINN trainer
///
/// Manages the training loop, optimizer, and learning rate scheduling.
///
/// # Type Parameters
///
/// * `B` - Autodiff backend (e.g., NdArray<f32, Autodiff>)
///
/// # Placeholder Status
///
/// This is a minimal implementation to allow compilation. The full training loop
/// with Burn optimizer integration is TODO.
#[cfg(feature = "pinn")]
pub struct Trainer<B: AutodiffBackend> {
    /// PINN model (non-autodiff version for inference)
    pub model: ElasticPINN2D<B::InnerBackend>,
    /// Training configuration
    pub config: Config,
    /// Loss function computer
    pub loss_computer: LossComputer,
    /// Training metrics
    pub metrics: TrainingMetrics,
}

#[cfg(feature = "pinn")]
impl<B: AutodiffBackend> Trainer<B> {
    /// Create a new trainer
    ///
    /// # Arguments
    ///
    /// * `model` - Initialized PINN model
    /// * `config` - Training configuration
    pub fn new(model: ElasticPINN2D<B::InnerBackend>, config: Config) -> Self {
        let loss_computer = LossComputer::new(config.loss_weights);

        Self {
            model,
            config,
            loss_computer,
            metrics: TrainingMetrics::new(),
        }
    }

    /// Train the PINN model (PLACEHOLDER)
    ///
    /// # Arguments
    ///
    /// * `_training_data` - Training data (collocation points, BCs, ICs, observations)
    ///
    /// # Returns
    ///
    /// Training metrics with loss history
    ///
    /// # Status
    ///
    /// This is a placeholder implementation. The full training loop requires:
    /// - Burn AutodiffModule integration
    /// - Proper gradient computation via backward()
    /// - Optimizer step with updated Burn 0.19+ API
    /// - Loss computation from model forward pass
    ///
    /// # TODO
    ///
    /// See phase4_action_plan.md Task 2 for implementation details.
    pub fn train(&mut self, _training_data: &TrainingData<B>) -> KwaversResult<TrainingMetrics> {
        let start_time = Instant::now();

        tracing::warn!(
            "PINN training is not yet implemented. This is a placeholder returning empty metrics."
        );
        tracing::warn!("See docs/phase4_action_plan.md Task 2 for implementation plan.");

        // Placeholder: return empty metrics
        self.metrics.total_time = start_time.elapsed().as_secs_f64();

        Ok(self.metrics.clone())
    }

    /// Get current model (for inference after training)
    pub fn model(&self) -> &ElasticPINN2D<B::InnerBackend> {
        &self.model
    }

    /// Get training metrics
    pub fn metrics(&self) -> &TrainingMetrics {
        &self.metrics
    }

    /// Check if training has converged
    pub fn has_converged(&self, tolerance: f64, window: usize) -> bool {
        self.metrics.has_converged(tolerance, window)
    }
}

// ============================================================================
// Helper Functions (Placeholders)
// ============================================================================

/// Create optimizer from configuration (PLACEHOLDER)
///
/// # TODO
///
/// Implement proper Burn 0.19+ optimizer initialization:
/// - Adam/AdamW with correct API
/// - SGD with momentum
/// - Learning rate scheduling integration
#[cfg(feature = "pinn")]
fn _create_optimizer_placeholder(config: &Config) -> KwaversResult<()> {
    match config.optimizer {
        super::config::OptimizerType::Adam { .. } => {
            let _adam = AdamConfig::new().init();
            Ok(())
        }
        super::config::OptimizerType::SGD { .. } => {
            let _sgd = SgdConfig::new().init();
            Ok(())
        }
        _ => Err(KwaversError::InvalidInput(
            "Optimizer type not yet implemented".to_string(),
        )),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_metrics_creation() {
        let metrics = TrainingMetrics::new();
        assert_eq!(metrics.epochs_completed, 0);
        assert!(metrics.total_loss.is_empty());
        assert_eq!(metrics.total_time, 0.0);
    }

    #[test]
    fn test_training_metrics_record() {
        let mut metrics = TrainingMetrics::new();
        metrics.record_epoch(1.0, 0.5, 0.3, 0.15, 0.05, 0.001, 1.5);

        assert_eq!(metrics.epochs_completed, 1);
        assert_eq!(metrics.total_loss.len(), 1);
        assert_eq!(metrics.total_loss[0], 1.0);
        assert_eq!(metrics.pde_loss[0], 0.5);
    }

    #[test]
    fn test_training_metrics_convergence() {
        let mut metrics = TrainingMetrics::new();

        // Not converged - loss decreasing
        for i in 0..10 {
            metrics.record_epoch(1.0 - i as f64 * 0.1, 0.5, 0.3, 0.15, 0.05, 0.001, 1.0);
        }
        assert!(!metrics.has_converged(0.01, 5));

        // Converged - loss stable
        for _ in 0..5 {
            metrics.record_epoch(0.001, 0.0005, 0.0003, 0.00015, 0.00005, 0.001, 1.0);
        }
        assert!(metrics.has_converged(0.01, 5));
    }

    #[test]
    fn test_training_metrics_final_loss() {
        let mut metrics = TrainingMetrics::new();
        assert!(metrics.final_loss().is_none());

        metrics.record_epoch(1.0, 0.5, 0.3, 0.15, 0.05, 0.001, 1.0);
        assert_eq!(metrics.final_loss(), Some(1.0));

        metrics.record_epoch(0.5, 0.25, 0.15, 0.075, 0.025, 0.001, 1.0);
        assert_eq!(metrics.final_loss(), Some(0.5));
    }

    #[test]
    fn test_training_metrics_average_epoch_time() {
        let mut metrics = TrainingMetrics::new();
        assert_eq!(metrics.average_epoch_time(), 0.0);

        metrics.record_epoch(1.0, 0.5, 0.3, 0.15, 0.05, 0.001, 1.0);
        metrics.record_epoch(0.8, 0.4, 0.25, 0.12, 0.03, 0.001, 1.5);
        metrics.record_epoch(0.6, 0.3, 0.2, 0.09, 0.01, 0.001, 2.0);

        let avg = metrics.average_epoch_time();
        assert!((avg - 1.5).abs() < 1e-10);
    }
}
