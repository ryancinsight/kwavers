//! `ConvergentBornStats` — convergence statistics for the CBS solver.

/// Statistics from Convergent Born Series solution.
#[derive(Debug, Clone, Default)]
pub struct ConvergentBornStats {
    /// Number of iterations performed.
    pub iterations: usize,
    /// Final residual value.
    pub final_residual: f64,
    /// Whether convergence was achieved.
    pub converged: bool,
}
