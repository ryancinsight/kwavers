//! Meta-learning performance statistics and convergence tracking.

/// Meta-learning performance statistics
///
/// Tracks cumulative statistics across meta-training to monitor:
/// - Training progress (epochs, tasks processed)
/// - Performance trends (average loss, best generalization)
/// - Convergence behavior (convergence rate)
#[derive(Debug, Clone)]
pub struct MetaLearningStats {
    /// Number of meta-training epochs completed
    pub meta_epochs_completed: usize,

    /// Total number of tasks processed across all epochs
    pub total_tasks_processed: usize,

    /// Average time per task adaptation (seconds)
    ///
    /// Used to estimate training time and identify bottlenecks.
    pub average_adaptation_time: f64,

    /// Exponential moving average of meta-loss
    ///
    /// Smoothed loss value for tracking trends.
    /// Update: EMA = α * new_loss + (1-α) * EMA, with α = 0.1
    pub average_meta_loss: f64,

    /// Best generalization score achieved so far
    ///
    /// Tracks the highest generalization score seen during training.
    /// Used for early stopping and model selection.
    pub best_generalization_score: f64,

    /// Convergence rate estimate
    ///
    /// Computed as: Δloss / Δepoch
    /// Negative values indicate loss is decreasing (good).
    /// Values near zero indicate convergence or plateau.
    pub convergence_rate: f64,
}

impl Default for MetaLearningStats {
    fn default() -> Self {
        Self {
            meta_epochs_completed: 0,
            total_tasks_processed: 0,
            average_adaptation_time: 0.0,
            average_meta_loss: 0.0,
            best_generalization_score: 0.0,
            convergence_rate: 0.0,
        }
    }
}

impl MetaLearningStats {
    /// Create new statistics tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Update statistics after a meta-training step
    ///
    /// # Arguments
    /// - `epochs_completed`: Total epochs completed so far
    /// - `tasks_in_batch`: Number of tasks in current batch
    /// - `meta_loss`: Current meta-loss value
    /// - `generalization_score`: Current generalization score
    pub fn update(
        &mut self,
        epochs_completed: usize,
        tasks_in_batch: usize,
        meta_loss: f64,
        generalization_score: f64,
    ) {
        // Update epoch count
        let prev_epochs = self.meta_epochs_completed;
        self.meta_epochs_completed = epochs_completed;

        // Update task count
        self.total_tasks_processed += tasks_in_batch;

        // Update average loss using exponential moving average (α = 0.1)
        let alpha = 0.1;
        if self.average_meta_loss == 0.0 {
            self.average_meta_loss = meta_loss;
        } else {
            self.average_meta_loss = alpha * meta_loss + (1.0 - alpha) * self.average_meta_loss;
        }

        // Update best generalization score
        if generalization_score > self.best_generalization_score {
            self.best_generalization_score = generalization_score;
        }

        // Estimate convergence rate (simplified)
        if epochs_completed > prev_epochs && prev_epochs > 0 {
            let delta_loss = meta_loss - self.average_meta_loss;
            let delta_epochs = (epochs_completed - prev_epochs) as f64;
            self.convergence_rate = delta_loss / delta_epochs;
        }
    }

    /// Update adaptation time statistics
    ///
    /// Uses exponential moving average to track average adaptation time.
    pub fn update_adaptation_time(&mut self, time_seconds: f64) {
        let alpha = 0.1;
        if self.average_adaptation_time == 0.0 {
            self.average_adaptation_time = time_seconds;
        } else {
            self.average_adaptation_time =
                alpha * time_seconds + (1.0 - alpha) * self.average_adaptation_time;
        }
    }

    /// Check if training is converging
    ///
    /// Returns true if:
    /// - Convergence rate is small (near zero or negative)
    /// - Best generalization score exceeds threshold
    pub fn is_converging(&self, rate_threshold: f64, gen_threshold: f64) -> bool {
        self.convergence_rate.abs() < rate_threshold
            && self.best_generalization_score > gen_threshold
    }

    /// Get average tasks per epoch
    pub fn tasks_per_epoch(&self) -> f64 {
        if self.meta_epochs_completed == 0 {
            0.0
        } else {
            self.total_tasks_processed as f64 / self.meta_epochs_completed as f64
        }
    }

    /// Get estimated total training time (seconds)
    pub fn estimated_total_time(&self) -> f64 {
        self.average_adaptation_time * self.total_tasks_processed as f64
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}
