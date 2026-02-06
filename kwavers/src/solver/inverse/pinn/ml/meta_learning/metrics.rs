//! Meta-Learning Metrics and Statistics
//!
//! This module defines performance metrics and statistics for meta-learning training,
//! including loss components, generalization scores, and convergence tracking.
//!
//! # Meta-Loss Components
//!
//! The meta-learning loss combines multiple objectives:
//! - **Task Loss**: Average loss across sampled tasks
//! - **Physics Loss**: PDE residual satisfaction
//! - **Generalization Score**: Variance across tasks (lower = better generalization)
//!
//! # Literature References
//!
//! 1. Finn, C., Abbeel, P., & Levine, S. (2017).
//!    "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"
//!    *ICML 2017*
//!    - Defines meta-learning loss as expected task loss after adaptation
//!
//! 2. Antoniou, A., Edwards, H., & Storkey, A. (2018).
//!    "How to train your MAML"
//!    *ICLR 2019*
//!    DOI: 10.48550/arXiv.1810.09502
//!    - Proposes improved training techniques and metrics
//!
//! 3. Raghu, A., et al. (2020).
//!    "Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML"
//!    *ICLR 2020*
//!    - Analyzes what makes meta-learning effective
//!
//! # Examples
//!
//! ```rust,ignore
//! use kwavers::solver::inverse::pinn::ml::meta_learning::{MetaLoss, MetaLearningStats};
//!
//! // Create a meta-loss result
//! let loss = MetaLoss {
//!     total_loss: 0.15,
//!     task_losses: vec![0.12, 0.18, 0.14, 0.16],
//!     physics_loss: 0.08,
//!     generalization_score: 0.92,
//! };
//!
//! // Check if meta-learning is generalizing well
//! if loss.generalization_score > 0.9 {
//!     println!("Good generalization across tasks!");
//! }
//!
//! // Track training statistics
//! let mut stats = MetaLearningStats::default();
//! stats.update(100, 8, 0.15, 0.92);
//! println!("Progress: {} epochs, {} tasks",
//!          stats.meta_epochs_completed, stats.total_tasks_processed);
//! ```

/// Meta-learning loss and metrics
///
/// Encapsulates the various loss components computed during meta-training.
/// The total loss guides meta-parameter updates, while individual components
/// provide diagnostic information about training progress.
///
/// # Mathematical Formulation
///
/// For MAML, the meta-objective is:
///
/// ```text
/// min_θ  𝔼_τ~p(τ) [ L_τ(U_τ(θ)) ]
/// ```
///
/// where:
/// - θ: meta-parameters (initial model weights)
/// - τ: task sampled from task distribution p(τ)
/// - U_τ(θ): adapted parameters after inner-loop updates on task τ
/// - L_τ: task-specific loss function
///
/// For Physics-Informed Neural Networks, the task loss L_τ includes:
///
/// ```text
/// L_τ = w_data * L_data + w_pde * L_pde + w_bc * L_bc + w_ic * L_ic
/// ```
///
/// where:
/// - L_data: Mean squared error on labeled data points
/// - L_pde: PDE residual at collocation points
/// - L_bc: Boundary condition violation
/// - L_ic: Initial condition violation
#[derive(Debug, Clone)]
pub struct MetaLoss {
    /// Total meta-loss across all tasks
    ///
    /// Average of task-specific losses after adaptation.
    /// This is the primary metric for meta-optimization.
    ///
    /// Range: [0, ∞), lower is better
    pub total_loss: f64,

    /// Task-specific losses
    ///
    /// Individual loss values for each task in the meta-batch.
    /// Used to compute variance and generalization metrics.
    ///
    /// Length: meta_batch_size
    pub task_losses: Vec<f64>,

    /// Physics constraint satisfaction loss
    ///
    /// Average PDE residual across all tasks.
    /// Measures how well the adapted models satisfy physics laws.
    ///
    /// Range: [0, ∞), lower is better
    /// Target: < 0.01 for good physics satisfaction
    pub physics_loss: f64,

    /// Generalization score across tasks
    ///
    /// Computed as: 1 / (1 + σ) where σ is standard deviation of task losses.
    /// Higher score indicates more consistent performance across tasks.
    ///
    /// Range: [0, 1], higher is better
    /// Target: > 0.8 for good generalization
    pub generalization_score: f64,
}

impl MetaLoss {
    /// Create a new meta-loss with computed generalization score
    pub fn new(task_losses: Vec<f64>, physics_loss: f64) -> Self {
        let total_loss = if !task_losses.is_empty() {
            task_losses.iter().sum::<f64>() / task_losses.len() as f64
        } else {
            0.0
        };

        let generalization_score = Self::compute_generalization(&task_losses);

        Self {
            total_loss,
            task_losses,
            physics_loss,
            generalization_score,
        }
    }

    /// Compute generalization score from task losses
    ///
    /// Uses coefficient of variation (CV) to measure consistency:
    /// - CV = σ / μ (standard deviation / mean)
    /// - Score = 1 / (1 + CV)
    ///
    /// This normalizes for different loss magnitudes and provides
    /// an intuitive 0-1 scale where 1 = perfect consistency.
    fn compute_generalization(task_losses: &[f64]) -> f64 {
        if task_losses.is_empty() {
            return 0.0;
        }

        let mean = task_losses.iter().sum::<f64>() / task_losses.len() as f64;

        if mean.abs() < 1e-10 {
            // If mean is near zero, return perfect score if all losses are near zero
            return if task_losses.iter().all(|&l| l.abs() < 1e-10) {
                1.0
            } else {
                0.0
            };
        }

        let variance = task_losses
            .iter()
            .map(|loss| (loss - mean).powi(2))
            .sum::<f64>()
            / task_losses.len() as f64;

        let std_dev = variance.sqrt();
        let cv = std_dev / mean.abs(); // Coefficient of variation

        // Convert to 0-1 score (higher = better)
        1.0 / (1.0 + cv)
    }

    /// Get the worst (highest) task loss
    pub fn worst_task_loss(&self) -> Option<f64> {
        self.task_losses
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get the best (lowest) task loss
    pub fn best_task_loss(&self) -> Option<f64> {
        self.task_losses
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get the standard deviation of task losses
    pub fn task_loss_std_dev(&self) -> f64 {
        if self.task_losses.is_empty() {
            return 0.0;
        }

        let mean = self.total_loss;
        let variance = self
            .task_losses
            .iter()
            .map(|loss| (loss - mean).powi(2))
            .sum::<f64>()
            / self.task_losses.len() as f64;

        variance.sqrt()
    }

    /// Check if meta-learning is converging well
    ///
    /// Returns true if:
    /// - Total loss is below threshold
    /// - Physics loss is below threshold
    /// - Generalization score is above threshold
    pub fn is_converged(
        &self,
        loss_threshold: f64,
        physics_threshold: f64,
        gen_threshold: f64,
    ) -> bool {
        self.total_loss < loss_threshold
            && self.physics_loss < physics_threshold
            && self.generalization_score > gen_threshold
    }
}

impl Default for MetaLoss {
    fn default() -> Self {
        Self {
            total_loss: 0.0,
            task_losses: Vec::new(),
            physics_loss: 0.0,
            generalization_score: 0.0,
        }
    }
}

/// Meta-learning performance statistics
///
/// Tracks cumulative statistics across meta-training to monitor:
/// - Training progress (epochs, tasks processed)
/// - Performance trends (average loss, best generalization)
/// - Convergence behavior (convergence rate)
///
/// # Usage in Training Loop
///
/// ```rust,ignore
/// let mut stats = MetaLearningStats::default();
///
/// for epoch in 0..config.meta_epochs {
///     let loss = meta_learner.meta_train_step()?;
///     stats.update(epoch + 1, loss.task_losses.len(),
///                  loss.total_loss, loss.generalization_score);
///
///     if epoch % 100 == 0 {
///         println!("Epoch {}: loss = {:.4}, gen = {:.3}",
///                  epoch, stats.average_meta_loss, stats.best_generalization_score);
///     }
/// }
/// ```
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_loss_default() {
        let loss = MetaLoss::default();
        assert_eq!(loss.total_loss, 0.0);
        assert_eq!(loss.physics_loss, 0.0);
        assert_eq!(loss.generalization_score, 0.0);
        assert!(loss.task_losses.is_empty());
    }

    #[test]
    fn test_meta_loss_new() {
        let task_losses = vec![0.1, 0.15, 0.12, 0.13];
        let physics_loss = 0.05;
        let loss = MetaLoss::new(task_losses.clone(), physics_loss);

        assert_eq!(loss.task_losses, task_losses);
        assert_eq!(loss.physics_loss, physics_loss);
        assert!((loss.total_loss - 0.125).abs() < 1e-10);
        assert!(loss.generalization_score > 0.0);
    }

    #[test]
    fn test_meta_loss_generalization_score() {
        // Perfect consistency (all same values)
        let perfect = vec![0.1, 0.1, 0.1, 0.1];
        let loss_perfect = MetaLoss::new(perfect, 0.0);
        assert!(loss_perfect.generalization_score > 0.99);

        // Poor consistency (high variance)
        let poor = vec![0.01, 0.1, 0.5, 1.0];
        let loss_poor = MetaLoss::new(poor, 0.0);
        assert!(loss_poor.generalization_score < 0.7);
    }

    #[test]
    fn test_meta_loss_worst_best_task() {
        let task_losses = vec![0.1, 0.5, 0.2, 0.3];
        let loss = MetaLoss::new(task_losses, 0.0);

        assert_eq!(loss.worst_task_loss(), Some(0.5));
        assert_eq!(loss.best_task_loss(), Some(0.1));
    }

    #[test]
    fn test_meta_loss_std_dev() {
        let task_losses = vec![0.1, 0.2, 0.3, 0.4];
        let loss = MetaLoss::new(task_losses, 0.0);

        let std = loss.task_loss_std_dev();
        assert!(std > 0.0);
        assert!(std < 0.2); // Should be reasonable
    }

    #[test]
    fn test_meta_loss_is_converged() {
        let task_losses = vec![0.01, 0.015, 0.012];
        let loss = MetaLoss::new(task_losses, 0.005);

        assert!(loss.is_converged(0.02, 0.01, 0.8));
        assert!(!loss.is_converged(0.01, 0.01, 0.8)); // Total loss too high
    }

    #[test]
    fn test_stats_default() {
        let stats = MetaLearningStats::default();
        assert_eq!(stats.meta_epochs_completed, 0);
        assert_eq!(stats.total_tasks_processed, 0);
        assert_eq!(stats.average_meta_loss, 0.0);
        assert_eq!(stats.best_generalization_score, 0.0);
    }

    #[test]
    fn test_stats_update() {
        let mut stats = MetaLearningStats::new();

        stats.update(1, 4, 0.5, 0.7);
        assert_eq!(stats.meta_epochs_completed, 1);
        assert_eq!(stats.total_tasks_processed, 4);
        assert_eq!(stats.average_meta_loss, 0.5);
        assert_eq!(stats.best_generalization_score, 0.7);

        stats.update(2, 4, 0.3, 0.8);
        assert_eq!(stats.meta_epochs_completed, 2);
        assert_eq!(stats.total_tasks_processed, 8);
        assert!(stats.best_generalization_score >= 0.8);
    }

    #[test]
    fn test_stats_adaptation_time() {
        let mut stats = MetaLearningStats::new();

        stats.update_adaptation_time(1.0);
        assert_eq!(stats.average_adaptation_time, 1.0);

        stats.update_adaptation_time(2.0);
        // Should be weighted average (closer to 1.0 due to α=0.1)
        assert!(stats.average_adaptation_time > 1.0);
        assert!(stats.average_adaptation_time < 1.5);
    }

    #[test]
    fn test_stats_tasks_per_epoch() {
        let mut stats = MetaLearningStats::new();
        stats.update(2, 8, 0.5, 0.7);
        stats.update(2, 8, 0.4, 0.8);

        assert_eq!(stats.tasks_per_epoch(), 8.0);
    }

    #[test]
    fn test_stats_is_converging() {
        let mut stats = MetaLearningStats::new();
        stats.convergence_rate = 0.001;
        stats.best_generalization_score = 0.85;

        assert!(stats.is_converging(0.01, 0.8));
        assert!(!stats.is_converging(0.0001, 0.8)); // Rate threshold too tight
        assert!(!stats.is_converging(0.01, 0.9)); // Gen threshold too high
    }

    #[test]
    fn test_stats_reset() {
        let mut stats = MetaLearningStats::new();
        stats.update(10, 40, 0.2, 0.9);

        stats.reset();
        assert_eq!(stats.meta_epochs_completed, 0);
        assert_eq!(stats.total_tasks_processed, 0);
        assert_eq!(stats.average_meta_loss, 0.0);
    }

    #[test]
    fn test_generalization_score_edge_cases() {
        // Empty task losses
        let empty = Vec::new();
        let loss_empty = MetaLoss::new(empty, 0.0);
        assert_eq!(loss_empty.generalization_score, 0.0);

        // Single task
        let single = vec![0.1];
        let loss_single = MetaLoss::new(single, 0.0);
        assert_eq!(loss_single.generalization_score, 1.0); // Perfect consistency

        // All zeros
        let zeros = vec![0.0, 0.0, 0.0];
        let loss_zeros = MetaLoss::new(zeros, 0.0);
        assert_eq!(loss_zeros.generalization_score, 1.0);
    }
}
