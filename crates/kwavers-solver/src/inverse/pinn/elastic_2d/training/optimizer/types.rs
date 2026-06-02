//! `OptimizerAlgorithm` — supported optimization algorithms for PINN training.

/// Supported optimization algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizerAlgorithm {
    /// Stochastic Gradient Descent
    SGD,
    /// SGD with momentum
    SGDMomentum,
    /// Adam optimizer
    Adam,
    /// Adam with decoupled weight decay
    AdamW,
}
