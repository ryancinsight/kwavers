//! Optimization algorithms for PINN training.
//!
//! Implements gradient descent optimizers with persistent state for
//! physics-informed neural network training.

pub mod pinn_optimizer;
pub mod types;

#[cfg(test)]
mod tests;

pub use types::OptimizerAlgorithm;

pub use pinn_optimizer::PINNOptimizer;
