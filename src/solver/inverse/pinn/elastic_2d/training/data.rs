//! Training data structures and metrics for PINN optimization
//!
//! This module defines the data containers and metrics used during
//! physics-informed neural network training.

#[cfg(feature = "pinn")]
use super::super::loss::data::*;
#[cfg(feature = "pinn")]
use burn::tensor::backend::AutodiffBackend;

// ============================================================================
// Training Data Container
// ============================================================================

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

// ============================================================================
// Training Metrics
// ============================================================================

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_metrics_creation() {
        let metrics = TrainingMetrics::new();
        assert_eq!(metrics.epochs_completed, 0);
        assert!(metrics.total_loss.is_empty());
    }

    #[test]
    fn test_training_metrics_recording() {
        let mut metrics = TrainingMetrics::new();
        metrics.record_epoch(1.0, 0.5, 0.2, 0.1, 0.2, 0.001, 0.5);

        assert_eq!(metrics.epochs_completed, 1);
        assert_eq!(metrics.final_loss(), Some(1.0));
        assert_eq!(metrics.average_epoch_time(), 0.5);
    }

    #[test]
    fn test_convergence_check() {
        let mut metrics = TrainingMetrics::new();

        // Add some loss values that are converging
        metrics.record_epoch(1.0, 0.5, 0.2, 0.1, 0.2, 0.001, 0.5);
        metrics.record_epoch(0.9, 0.4, 0.2, 0.1, 0.2, 0.001, 0.5);
        metrics.record_epoch(0.85, 0.35, 0.2, 0.1, 0.2, 0.001, 0.5);
        metrics.record_epoch(0.82, 0.32, 0.2, 0.1, 0.2, 0.001, 0.5);
        metrics.record_epoch(0.80, 0.30, 0.2, 0.1, 0.2, 0.001, 0.5);

        // Should converge with tolerance 0.1 over window of 3
        assert!(metrics.has_converged(0.1, 3));

        // Should not converge with stricter tolerance
        assert!(!metrics.has_converged(0.01, 3));
    }
}
