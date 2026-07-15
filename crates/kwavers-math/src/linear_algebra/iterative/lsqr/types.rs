//! LSQR configuration and result types.

use leto::Array1;

/// Configuration for LSQR solver
#[derive(Debug, Clone, Copy)]
pub struct LsqrConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance (relative residual)
    pub tolerance: f64,
    /// Tikhonov damping parameter λ ≥ 0: minimise ‖Ax − b‖² + λ²‖x‖²
    pub damping: f64,
    /// Tolerance on `‖Aᵀr‖` (normal-equation residual)
    pub atol: f64,
    /// Tolerance on `‖r‖` (residual)
    pub btol: f64,
}

impl Default for LsqrConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            damping: 0.0,
            atol: 1e-8,
            btol: 1e-8,
        }
    }
}

/// Results from LSQR solver
#[derive(Debug, Clone)]
pub struct LsqrResult {
    /// Solution vector x
    pub solution: Array1<f64>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final residual norm ‖Ax − b‖ estimate
    pub residual_norm: f64,
    /// Final normal-equation residual ‖Aᵀ(Ax − b)‖ estimate
    pub at_residual_norm: f64,
    /// Condition number estimate of A
    pub condition_number: f64,
    /// True if a stopping criterion was satisfied before `max_iterations`
    pub converged: bool,
    /// Stopping reason
    pub stop_reason: StopReason,
}

/// Reason for stopping iteration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Converged to solution
    Converged,
    /// Maximum iterations reached
    MaxIterations,
    /// `‖Aᵀr‖` tolerance satisfied
    AtolSatisfied,
    /// `‖r‖` tolerance satisfied
    BtolSatisfied,
    /// Condition number too large (singular or near-singular matrix)
    IllConditioned,
}
