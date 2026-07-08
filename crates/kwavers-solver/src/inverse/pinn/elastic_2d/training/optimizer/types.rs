//! `OptimizerAlgorithm` — supported optimization algorithms for PINN training.

/// Supported optimization algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizerAlgorithm {
    /// Stochastic Gradient Descent.
    SGD,
    /// SGD with a momentum-scaled step.
    SGDMomentum,
    /// Adam (Coeus optimizer: full first/second-moment update + bias correction).
    Adam,
    /// AdamW (Adam with decoupled weight decay; Coeus optimizer).
    AdamW,
}
