//! Training history and convergence tracking.

use super::dataset::TrainingMetrics;

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
        improvement / self.epochs.len() as f64
    }
}

impl Default for TrainingHistory {
    fn default() -> Self {
        Self::new()
    }
}
