//! Convergence information type for GMRES.

/// Convergence information returned by a GMRES solve.
#[derive(Debug, Clone)]
pub struct GmresConvergenceInfo {
    /// Whether solver converged within tolerance.
    pub converged: bool,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Final residual norm ||b - A·x||.
    pub final_residual: f64,
    /// Relative residual ||r|| / ||b||.
    pub relative_residual: f64,
}
