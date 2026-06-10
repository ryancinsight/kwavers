//! `OptimizerAlgorithm` — supported optimization algorithms for PINN training.

/// Supported optimization algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizerAlgorithm {
    /// Stochastic Gradient Descent.
    SGD,
    /// SGD with a momentum-scaled step.
    SGDMomentum,
    /// Adam (burn's built-in: full first/second-moment update + bias correction).
    Adam,
    /// AdamW (Adam with decoupled weight decay; burn's built-in).
    AdamW,
}
