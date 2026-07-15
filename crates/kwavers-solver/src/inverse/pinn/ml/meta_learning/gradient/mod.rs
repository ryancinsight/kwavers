//! Gradient Manipulation Utilities for Meta-Learning
//!
//! This module provides utilities for manipulating flat per-parameter gradient
//! snapshots, essential for implementing Model-Agnostic Meta-Learning (MAML)
//! algorithms.
//!
//! # Gradient Flow in MAML
//!
//! MAML requires manipulating gradients at two levels:
//! 1. **Inner Loop**: Task-specific gradient descent for adaptation
//! 2. **Outer Loop**: Meta-gradient computation across task distribution
//!
//! Task-level gradients are extracted directly via `coeus_autograd::Var::grad()`
//! on each of the model's `parameters()` (see `MetaLearner::compute_gradients_and_loss`
//! and `MetaLearner::meta_train_step`), read out to plain `Vec<f32>` snapshots
//! (one per parameter, in `parameters()` order), and re-applied via
//! `coeus_nn::Module::load_parameters`. The utilities below operate on that
//! flat `Vec<Option<Vec<f32>>>` representation.
//!
//! # Literature References
//!
//! 1. Finn, C., Abbeel, P., & Levine, S. (2017).
//!    "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"
//!    *ICML 2017*
//!    - Original MAML algorithm requiring manual gradient manipulation
//!
//! 2. Nichol, A., Achiam, J., & Schulman, J. (2018).
//!    "On First-Order Meta-Learning Algorithms"
//!    *arXiv:1803.02999*
//!    - First-order approximation simplifying gradient computation
//!
//! 3. Antoniou, A., Edwards, H., & Storkey, A. (2018).
//!    "How to train your MAML"
//!    *ICLR 2019*
//!    - Practical improvements including gradient clipping and normalization

#[cfg(test)]
mod tests;

/// Utility functions for gradient manipulation
pub mod utils;
