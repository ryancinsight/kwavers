//! Domain Types for Meta-Learning
//!
//! Defines core domain types for meta-learning with Physics-Informed Neural Networks,
//! including task definitions, physics parameters, and training data structures.
//!
//! # Literature References
//!
//! 1. Finn, C., Abbeel, P., & Levine, S. (2017).
//!    "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" ICML 2017
//!
//! 2. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).
//!    "Physics-informed neural networks" Journal of Computational Physics, 378, 686-707.
//!    DOI: 10.1016/j.jcp.2018.10.045

pub mod pde_type;
pub mod physics;
pub mod task;
#[cfg(test)]
mod tests;

pub use pde_type::PdeType;
pub use physics::PhysicsParameters;
pub use task::{PhysicsTask, TaskData, TaskDataStatistics};
