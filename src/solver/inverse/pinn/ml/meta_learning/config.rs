//! Meta-Learning Configuration
//!
//! Configuration parameters for Model-Agnostic Meta-Learning (MAML) training
//! of Physics-Informed Neural Networks.
//!
//! # Configuration Parameters
//!
//! ## Learning Rates
//! - **Inner Loop (`inner_lr`)**: Learning rate for task-specific adaptation
//!   - Typical range: 0.001 - 0.1
//!   - Controls how quickly the model adapts to individual tasks
//!   - Literature: Finn et al. (2017) "Model-Agnostic Meta-Learning for Fast Adaptation"
//!
//! - **Outer Loop (`outer_lr`)**: Learning rate for meta-parameter updates
//!   - Typical range: 0.0001 - 0.01
//!   - Controls meta-learning across task distribution
//!   - Usually 10-100x smaller than inner_lr
//!
//! ## Adaptation Parameters
//! - **Adaptation Steps**: Number of gradient steps during task adaptation
//!   - Typical range: 1-10 steps
//!   - More steps = better task adaptation but higher computational cost
//!   - MAML paper uses 1-5 steps typically
//!
//! ## Meta-Training Parameters
//! - **Meta Batch Size**: Number of tasks sampled per meta-update
//!   - Typical range: 4-32 tasks
//!   - Trade-off between gradient variance and computation
//!
//! - **Meta Epochs**: Number of complete passes through task distribution
//!   - Typical range: 100-10000 epochs
//!   - Depends on task complexity and diversity
//!
//! ## Algorithm Variants
//! - **First-Order MAML (`first_order`)**: Use first-order approximation
//!   - Ignores second-order derivatives (faster, less accurate)
//!   - Reptile algorithm is equivalent to first-order MAML with multiple steps
//!   - Literature: Nichol et al. (2018) "On First-Order Meta-Learning Algorithms"
//!
//! ## Physics Regularization
//! - **Physics Regularization Weight**: Strength of physics constraint enforcement
//!   - Typical range: 0.01 - 1.0
//!   - Balances data fitting vs. physics satisfaction
//!
//! # Literature References
//!
//! 1. Finn, C., Abbeel, P., & Levine, S. (2017).
//!    "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"
//!    *ICML 2017*
//!    DOI: 10.5555/3305381.3305498
//!
//! 2. Nichol, A., Achiam, J., & Schulman, J. (2018).
//!    "On First-Order Meta-Learning Algorithms"
//!    *arXiv:1803.02999*
//!
//! 3. Antoniou, A., Edwards, H., & Storkey, A. (2018).
//!    "How to train your MAML"
//!    *ICLR 2019*
//!    DOI: 10.48550/arXiv.1810.09502
//!
//! # Examples
//!
//! ```rust,ignore
//! use kwavers::solver::inverse::pinn::ml::meta_learning::MetaLearningConfig;
//!
//! // Standard MAML configuration for wave equation tasks
//! let config = MetaLearningConfig {
//!     inner_lr: 0.01,           // Fast adaptation rate
//!     outer_lr: 0.001,          // Meta-learning rate
//!     adaptation_steps: 5,       // 5 gradient steps per task
//!     meta_batch_size: 8,        // 8 tasks per meta-update
//!     meta_epochs: 1000,         // 1000 meta-training epochs
//!     first_order: false,        // Use full second-order MAML
//!     physics_regularization: 0.1,
//!     num_layers: 4,
//!     hidden_dim: 64,
//!     input_dim: 3,              // (x, y, t)
//!     output_dim: 1,             // u(x, y, t)
//!     max_tasks: 100,
//! };
//!
//! // First-order MAML (Reptile-like) for faster training
//! let fast_config = MetaLearningConfig {
//!     first_order: true,
//!     adaptation_steps: 10,      // More steps to compensate
//!     ..config
//! };
//! ```

/// Meta-learning configuration for MAML training
///
/// Specifies hyperparameters for both inner-loop (task adaptation)
/// and outer-loop (meta-learning) optimization.
#[derive(Debug, Clone)]
pub struct MetaLearningConfig {
    /// Inner-loop learning rate for task adaptation
    ///
    /// Controls how quickly the model adapts to individual tasks during
    /// the inner optimization loop. Typical values: 0.001 - 0.1
    pub inner_lr: f64,

    /// Outer-loop learning rate for meta-parameter updates
    ///
    /// Controls the meta-learning rate across the task distribution.
    /// Usually 10-100x smaller than `inner_lr`. Typical values: 0.0001 - 0.01
    pub outer_lr: f64,

    /// Number of inner-loop adaptation steps
    ///
    /// Number of gradient descent steps performed during task-specific
    /// adaptation. More steps improve adaptation but increase computation.
    /// Typical values: 1-10 steps
    pub adaptation_steps: usize,

    /// Number of tasks per meta-batch
    ///
    /// Number of tasks sampled for each meta-gradient computation.
    /// Larger batches reduce gradient variance but require more memory.
    /// Typical values: 4-32 tasks
    pub meta_batch_size: usize,

    /// Meta-training epochs
    ///
    /// Number of complete passes through the task distribution.
    /// Depends on task complexity and diversity.
    /// Typical values: 100-10000 epochs
    pub meta_epochs: usize,

    /// First-order approximation (FO-MAML)
    ///
    /// If true, uses first-order approximation ignoring second-order derivatives.
    /// Faster but potentially less accurate than full MAML.
    /// Reptile algorithm is equivalent to FO-MAML with multiple adaptation steps.
    pub first_order: bool,

    /// Physics-aware regularization weight
    ///
    /// Strength of physics constraint enforcement in the loss function.
    /// Balances data fitting vs. PDE satisfaction.
    /// Typical values: 0.01 - 1.0
    pub physics_regularization: f64,

    /// Number of hidden layers in PINN architecture
    ///
    /// Depth of the neural network. More layers increase expressiveness
    /// but also training difficulty.
    /// Typical values: 3-8 layers
    pub num_layers: usize,

    /// Hidden layer dimension
    ///
    /// Width of each hidden layer. Larger dimensions increase capacity.
    /// Typical values: 20-256 neurons
    pub hidden_dim: usize,

    /// Input dimension
    ///
    /// Dimensionality of input space (e.g., 3 for (x, y, t)).
    pub input_dim: usize,

    /// Output dimension
    ///
    /// Dimensionality of output space (e.g., 1 for scalar field u).
    pub output_dim: usize,

    /// Maximum number of tasks for curriculum learning
    ///
    /// Upper bound on task pool size for curriculum-based sampling strategies.
    /// Used to compute learning progress ratios.
    pub max_tasks: usize,
}

impl Default for MetaLearningConfig {
    fn default() -> Self {
        Self {
            inner_lr: 0.01,
            outer_lr: 0.001,
            adaptation_steps: 5,
            meta_batch_size: 8,
            meta_epochs: 1000,
            first_order: false,
            physics_regularization: 0.1,
            num_layers: 4,
            hidden_dim: 64,
            input_dim: 3,
            output_dim: 1,
            max_tasks: 100,
        }
    }
}

impl MetaLearningConfig {
    /// Create a new configuration with validation
    ///
    /// # Validation Rules
    /// - Learning rates must be positive
    /// - Adaptation steps must be at least 1
    /// - Meta batch size must be at least 1
    /// - Meta epochs must be at least 1
    /// - Network dimensions must be at least 1
    ///
    /// # Errors
    /// Returns error if any validation rule is violated
    pub fn new(
        inner_lr: f64,
        outer_lr: f64,
        adaptation_steps: usize,
        meta_batch_size: usize,
        meta_epochs: usize,
        first_order: bool,
        physics_regularization: f64,
        num_layers: usize,
        hidden_dim: usize,
        input_dim: usize,
        output_dim: usize,
        max_tasks: usize,
    ) -> Result<Self, String> {
        // Validate learning rates
        if inner_lr <= 0.0 {
            return Err(format!("inner_lr must be positive, got {}", inner_lr));
        }
        if outer_lr <= 0.0 {
            return Err(format!("outer_lr must be positive, got {}", outer_lr));
        }

        // Validate training parameters
        if adaptation_steps == 0 {
            return Err("adaptation_steps must be at least 1".to_string());
        }
        if meta_batch_size == 0 {
            return Err("meta_batch_size must be at least 1".to_string());
        }
        if meta_epochs == 0 {
            return Err("meta_epochs must be at least 1".to_string());
        }

        // Validate physics regularization
        if physics_regularization < 0.0 {
            return Err(format!(
                "physics_regularization must be non-negative, got {}",
                physics_regularization
            ));
        }

        // Validate network architecture
        if num_layers == 0 {
            return Err("num_layers must be at least 1".to_string());
        }
        if hidden_dim == 0 {
            return Err("hidden_dim must be at least 1".to_string());
        }
        if input_dim == 0 {
            return Err("input_dim must be at least 1".to_string());
        }
        if output_dim == 0 {
            return Err("output_dim must be at least 1".to_string());
        }
        if max_tasks == 0 {
            return Err("max_tasks must be at least 1".to_string());
        }

        Ok(Self {
            inner_lr,
            outer_lr,
            adaptation_steps,
            meta_batch_size,
            meta_epochs,
            first_order,
            physics_regularization,
            num_layers,
            hidden_dim,
            input_dim,
            output_dim,
            max_tasks,
        })
    }

    /// Create configuration for fast prototyping (minimal adaptation)
    pub fn fast() -> Self {
        Self {
            adaptation_steps: 1,
            meta_batch_size: 4,
            first_order: true,
            ..Default::default()
        }
    }

    /// Create configuration for high-quality meta-learning (extensive adaptation)
    pub fn high_quality() -> Self {
        Self {
            adaptation_steps: 10,
            meta_batch_size: 16,
            meta_epochs: 5000,
            first_order: false,
            ..Default::default()
        }
    }

    /// Create configuration for large-scale problems
    pub fn large_scale() -> Self {
        Self {
            num_layers: 8,
            hidden_dim: 256,
            adaptation_steps: 5,
            meta_batch_size: 32,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MetaLearningConfig::default();
        assert_eq!(config.inner_lr, 0.01);
        assert_eq!(config.outer_lr, 0.001);
        assert_eq!(config.adaptation_steps, 5);
        assert_eq!(config.meta_batch_size, 8);
        assert_eq!(config.meta_epochs, 1000);
        assert!(!config.first_order);
        assert_eq!(config.physics_regularization, 0.1);
        assert_eq!(config.num_layers, 4);
        assert_eq!(config.hidden_dim, 64);
        assert_eq!(config.input_dim, 3);
        assert_eq!(config.output_dim, 1);
        assert_eq!(config.max_tasks, 100);
    }

    #[test]
    fn test_new_config_valid() {
        let config =
            MetaLearningConfig::new(0.01, 0.001, 5, 8, 1000, false, 0.1, 4, 64, 3, 1, 100).unwrap();
        assert_eq!(config.inner_lr, 0.01);
        assert_eq!(config.adaptation_steps, 5);
    }

    #[test]
    fn test_new_config_invalid_inner_lr() {
        let result =
            MetaLearningConfig::new(-0.01, 0.001, 5, 8, 1000, false, 0.1, 4, 64, 3, 1, 100);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("inner_lr"));
    }

    #[test]
    fn test_new_config_invalid_outer_lr() {
        let result = MetaLearningConfig::new(0.01, 0.0, 5, 8, 1000, false, 0.1, 4, 64, 3, 1, 100);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("outer_lr"));
    }

    #[test]
    fn test_new_config_invalid_adaptation_steps() {
        let result = MetaLearningConfig::new(0.01, 0.001, 0, 8, 1000, false, 0.1, 4, 64, 3, 1, 100);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("adaptation_steps"));
    }

    #[test]
    fn test_new_config_invalid_meta_batch_size() {
        let result = MetaLearningConfig::new(0.01, 0.001, 5, 0, 1000, false, 0.1, 4, 64, 3, 1, 100);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("meta_batch_size"));
    }

    #[test]
    fn test_new_config_invalid_physics_regularization() {
        let result =
            MetaLearningConfig::new(0.01, 0.001, 5, 8, 1000, false, -0.1, 4, 64, 3, 1, 100);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("physics_regularization"));
    }

    #[test]
    fn test_fast_config() {
        let config = MetaLearningConfig::fast();
        assert_eq!(config.adaptation_steps, 1);
        assert_eq!(config.meta_batch_size, 4);
        assert!(config.first_order);
    }

    #[test]
    fn test_high_quality_config() {
        let config = MetaLearningConfig::high_quality();
        assert_eq!(config.adaptation_steps, 10);
        assert_eq!(config.meta_batch_size, 16);
        assert_eq!(config.meta_epochs, 5000);
        assert!(!config.first_order);
    }

    #[test]
    fn test_large_scale_config() {
        let config = MetaLearningConfig::large_scale();
        assert_eq!(config.num_layers, 8);
        assert_eq!(config.hidden_dim, 256);
        assert_eq!(config.meta_batch_size, 32);
    }
}
