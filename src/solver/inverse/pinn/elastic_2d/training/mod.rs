//! Training loop and optimization for 2D Elastic Wave PINN
//!
//! This module implements the complete training procedure for physics-informed neural networks
//! solving the 2D elastic wave equation.
//!
//! # Mathematical Foundation
//!
//! The training minimizes the physics-informed loss:
//!
//! θ* = arg min_θ L(θ) = arg min_θ [w_pde·L_pde + w_bc·L_bc + w_ic·L_ic + w_data·L_data]
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
//! θ_{k+1} = θ_k - α_k ∇_θ L(θ_k)
//!
//! where α_k is the learning rate at iteration k (possibly scheduled).

// Re-export submodules
pub mod data;
pub mod r#loop;
pub mod optimizer;
pub mod scheduler;

// ============================================================================
// EXPLICIT RE-EXPORTS (PINN Training API)
// ============================================================================

/// Training data and metrics
pub use data::{TrainingData, TrainingMetrics};

/// Optimization algorithms and state
pub use optimizer::{OptimizerAlgorithm, PINNOptimizer, PersistentAdamState};

/// Learning rate scheduling
pub use scheduler::{LRScheduler, LearningRateScheduler};
