//! Optimization algorithms for PINN training.
//!
//! Implements gradient descent optimizers with persistent state for
//! physics-informed neural network training.

pub mod mappers;
pub mod pinn_optimizer;
pub mod state;
pub mod types;

#[cfg(test)]
mod tests;

pub use types::OptimizerAlgorithm;

#[cfg(feature = "pinn")]
pub use pinn_optimizer::PINNOptimizer;
#[cfg(feature = "pinn")]
pub use state::{MomentumState, PersistentAdamState};
