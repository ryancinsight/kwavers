//! Meta-Learning Framework for Physics-Informed Neural Networks
//!
//! This module implements Model-Agnostic Meta-Learning (MAML) for Physics-Informed
//! Neural Networks, enabling fast adaptation to new physics problems and geometries
//! through learned optimal initialization.
//!
//! # Overview
//!
//! Meta-learning, or "learning to learn," trains models to quickly adapt to new tasks
//! with minimal data. For Physics-Informed Neural Networks (PINNs), meta-learning enables:
//!
//! - **Fast Adaptation**: Quickly solve new PDE problems with few gradient steps
//! - **Transfer Learning**: Leverage knowledge across related physics domains
//! - **Few-Shot Learning**: Adapt with minimal labeled data or measurements
//! - **Robust Initialization**: Find parameter initializations that generalize well
//!
//! # MAML Algorithm
//!
//! Model-Agnostic Meta-Learning operates in two loops:
//!
//! ## Inner Loop (Task Adaptation)
//! ```text
//! For each task τ:
//!     θ'_τ = θ - α∇_θ L_τ(θ)  // Adapt to task τ
//! ```
//!
//! ## Outer Loop (Meta-Optimization)
//! ```text
//! θ = θ - β∇_θ Σ_τ L_τ(θ'_τ)  // Update meta-parameters
//! ```
//!
//! where:
//! - θ: meta-parameters (initial model weights)
//! - α: inner-loop learning rate (task adaptation)
//! - β: outer-loop learning rate (meta-learning)
//! - L_τ: task-specific loss function
//!
//! # Module Structure
//!
//! - **[`config`]**: Configuration parameters for meta-learning
//! - **[`types`]**: Domain types (tasks, physics parameters, data)
//! - **[`metrics`]**: Loss components and performance statistics
//! - **[`gradient`]**: Gradient manipulation utilities for Burn framework
//! - **[`optimizer`]**: Meta-optimizer for outer-loop updates
//! - **[`sampling`]**: Task sampling strategies (curriculum, diversity)
//! - **[`learner`]**: Core MAML implementation
//!
//! # Literature References
//!
//! 1. **Finn, C., Abbeel, P., & Levine, S. (2017).**
//!    "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"
//!    *ICML 2017*
//!    DOI: 10.5555/3305381.3305498
//!
//! 2. **Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).**
//!    "Physics-informed neural networks: A deep learning framework for solving
//!    forward and inverse problems involving nonlinear partial differential equations"
//!    *Journal of Computational Physics*, 378, 686-707.
//!    DOI: 10.1016/j.jcp.2018.10.045
//!
//! 3. **Nichol, A., Achiam, J., & Schulman, J. (2018).**
//!    "On First-Order Meta-Learning Algorithms"
//!    *arXiv:1803.02999*
//!
//! 4. **Antoniou, A., Edwards, H., & Storkey, A. (2018).**
//!    "How to train your MAML"
//!    *ICLR 2019*
//!    DOI: 10.48550/arXiv.1810.09502
//!
//! 5. **Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009).**
//!    "Curriculum Learning"
//!    *ICML 2009*
//!
//! # Examples
//!
//! ## Basic Meta-Learning Setup
//!
//! ```rust,ignore
//! use kwavers::analysis::ml::pinn::meta_learning::*;
//! use burn::backend::Autodiff;
//!
//! // Configure meta-learning
//! let config = MetaLearningConfig {
//!     inner_lr: 0.01,           // Task adaptation rate
//!     outer_lr: 0.001,          // Meta-learning rate
//!     adaptation_steps: 5,       // Gradient steps per task
//!     meta_batch_size: 8,        // Tasks per meta-update
//!     meta_epochs: 1000,         // Total meta-training epochs
//!     first_order: false,        // Use full second-order MAML
//!     physics_regularization: 0.1,
//!     num_layers: 4,
//!     hidden_dim: 64,
//!     input_dim: 3,              // (x, y, t)
//!     output_dim: 1,             // u(x, y, t)
//!     max_tasks: 100,
//! };
//!
//! // Create meta-learner
//! let mut meta_learner = MetaLearner::new(config, &device)?;
//!
//! // Add tasks to task pool
//! for task in task_distribution {
//!     meta_learner.add_task(task);
//! }
//!
//! // Meta-training loop
//! for epoch in 0..config.meta_epochs {
//!     let loss = meta_learner.meta_train_step()?;
//!
//!     if epoch % 100 == 0 {
//!         println!("Epoch {}: loss = {:.4}, gen = {:.3}",
//!                  epoch, loss.total_loss, loss.generalization_score);
//!     }
//! }
//! ```
//!
//! ## Creating Physics Tasks
//!
//! ```rust,ignore
//! use kwavers::analysis::ml::pinn::meta_learning::*;
//! use std::sync::Arc;
//!
//! // Define physics parameters for acoustic wave
//! let params = PhysicsParameters {
//!     wave_speed: 343.0,  // Speed of sound in air
//!     density: 1.2,       // Air density
//!     viscosity: None,
//!     absorption: Some(0.01),
//!     nonlinearity: None,
//! };
//!
//! // Create a task
//! let task = PhysicsTask {
//!     id: "acoustic_wave_rect_1".to_string(),
//!     pde_type: PdeType::Wave,
//!     physics_params: params,
//!     geometry: Arc::new(Geometry2D::rectangular(0.0, 1.0, 0.0, 1.0)),
//!     boundary_conditions: vec![],
//!     training_data: None,
//!     validation_data: TaskData {
//!         collocation_points: generate_collocation_points(),
//!         boundary_data: generate_boundary_data(),
//!         initial_data: generate_initial_data(),
//!     },
//! };
//! ```
//!
//! ## Task Sampling Strategies
//!
//! ```rust,ignore
//! use kwavers::analysis::ml::pinn::meta_learning::*;
//!
//! // Random sampling
//! let sampler = TaskSampler::new(SamplingStrategy::Random, config.clone());
//!
//! // Curriculum learning (easy to hard)
//! let sampler = TaskSampler::new(SamplingStrategy::Curriculum, config.clone());
//!
//! // Diversity-based sampling
//! let sampler = TaskSampler::new(SamplingStrategy::Diversity, config.clone());
//! ```
//!
//! # Performance Considerations
//!
//! ## Computational Cost
//! - **Inner Loop**: O(K × N) where K = adaptation steps, N = model parameters
//! - **Outer Loop**: O(M × K × N) where M = meta-batch size
//! - **Memory**: Requires storing M copies of model for task adaptation
//!
//! ## Optimization Tips
//! 1. **First-Order MAML**: Set `first_order: true` for 2x speedup
//! 2. **Batch Size**: Start with 4-8 tasks, increase if memory allows
//! 3. **Adaptation Steps**: 1-5 steps usually sufficient for fast adaptation
//! 4. **Learning Rates**: outer_lr typically 10-100x smaller than inner_lr
//!
//! # Current Limitations
//!
//! - Simplified gradient computation using finite differences
//! - Limited to 2D wave equations in current PINN implementation
//! - Manual gradient manipulation (Burn limitation)
//! - Memory inefficient for large models (copies model M times)
//!
//! # Future Improvements
//!
//! - [ ] Full automatic differentiation for meta-gradients
//! - [ ] Support for arbitrary neural network architectures
//! - [ ] Multi-physics coupling scenarios
//! - [ ] Adaptive learning rate scheduling
//! - [ ] Gradient checkpointing for memory efficiency
//! - [ ] Distributed meta-training across multiple GPUs

// Module declarations
pub mod config;
pub mod gradient;
pub mod learner;
pub mod metrics;
pub mod optimizer;
pub mod sampling;
pub mod types;

// Public API re-exports
pub use config::MetaLearningConfig;
pub use gradient::{GradientApplicator, GradientExtractor};
pub use learner::MetaLearner;
pub use metrics::{MetaLearningStats, MetaLoss};
pub use optimizer::{LearningRateSchedule, MetaOptimizer};
pub use sampling::{SamplingStrategy, TaskSampler};
pub use types::{PdeType, PhysicsParameters, PhysicsTask, TaskData, TaskDataStatistics};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify all types are accessible
        let _config = MetaLearningConfig::default();
        let _sampling_strategy = SamplingStrategy::Random;
        let _pde_type = PdeType::Wave;
        let _params = PhysicsParameters::default();
        let _data = TaskData::default();
        let _stats = MetaLearningStats::default();
        let _loss = MetaLoss::default();
    }

    #[test]
    fn test_config_presets() {
        let fast = MetaLearningConfig::fast();
        assert_eq!(fast.adaptation_steps, 1);
        assert!(fast.first_order);

        let high_quality = MetaLearningConfig::high_quality();
        assert_eq!(high_quality.adaptation_steps, 10);
        assert!(!high_quality.first_order);

        let large_scale = MetaLearningConfig::large_scale();
        assert_eq!(large_scale.num_layers, 8);
        assert_eq!(large_scale.hidden_dim, 256);
    }

    #[test]
    fn test_pde_complexity_ordering() {
        assert!(PdeType::Wave.complexity() < PdeType::Diffusion.complexity());
        assert!(PdeType::Diffusion.complexity() < PdeType::NavierStokes.complexity());
        assert!(PdeType::NavierStokes.complexity() == 1.0);
    }

    #[test]
    fn test_physics_parameters_presets() {
        let air = PhysicsParameters::acoustic_air();
        assert_eq!(air.wave_speed, 343.0);

        let water = PhysicsParameters::acoustic_water();
        assert_eq!(water.wave_speed, 1500.0);

        let tissue = PhysicsParameters::acoustic_tissue();
        assert_eq!(tissue.wave_speed, 1540.0);
        assert!(tissue.nonlinearity.is_some());
    }

    #[test]
    fn test_meta_loss_generalization() {
        // Test perfect consistency
        let perfect_losses = vec![0.1, 0.1, 0.1, 0.1];
        let loss_perfect = MetaLoss::new(perfect_losses, 0.05);
        assert!(loss_perfect.generalization_score > 0.99);

        // Test poor consistency
        let poor_losses = vec![0.01, 0.1, 0.5, 1.0];
        let loss_poor = MetaLoss::new(poor_losses, 0.05);
        assert!(loss_poor.generalization_score < 0.7);
    }

    #[test]
    fn test_learning_rate_schedules() {
        let constant = LearningRateSchedule::Constant;
        assert_eq!(constant.get_lr(0, 0.001), 0.001);
        assert_eq!(constant.get_lr(1000, 0.001), 0.001);

        let step_decay = LearningRateSchedule::StepDecay {
            factor: 0.5,
            step_size: 100,
        };
        assert_eq!(step_decay.get_lr(0, 0.001), 0.001);
        assert_eq!(step_decay.get_lr(100, 0.001), 0.0005);

        let exponential = LearningRateSchedule::Exponential { decay_rate: 0.01 };
        let lr_start = exponential.get_lr(0, 0.001);
        let lr_later = exponential.get_lr(100, 0.001);
        assert!(lr_later < lr_start);
    }
}
