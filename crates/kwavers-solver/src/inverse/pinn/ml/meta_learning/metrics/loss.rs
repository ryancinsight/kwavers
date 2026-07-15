//! Meta-learning loss and generalization metrics.
//!
//! # Literature References
//!
//! 1. Finn, C., Abbeel, P., & Levine, S. (2017).
//!    "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"
//!    *ICML 2017*
//!
//! 2. Antoniou, A., Edwards, H., & Storkey, A. (2018).
//!    "How to train your MAML" *ICLR 2019*
//!    DOI: 10.48550/arXiv.1810.09502

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
            task_losses.iter().sum::<f64>() / (task_losses.len()) as f64
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

        let mean = task_losses.iter().sum::<f64>() / (task_losses.len()) as f64;

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
            / (task_losses.len()) as f64;

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
            .max_by(|a, b| a.total_cmp(b))
    }

    /// Get the best (lowest) task loss
    pub fn best_task_loss(&self) -> Option<f64> {
        self.task_losses
            .iter()
            .copied()
            .min_by(|a, b| a.total_cmp(b))
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
            / (self.task_losses.len()) as f64;

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
