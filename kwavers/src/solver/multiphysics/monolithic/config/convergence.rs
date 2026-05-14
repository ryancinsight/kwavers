/// Coupling convergence information for one monolithic solve step.
#[derive(Debug, Clone)]
pub struct CouplingConvergenceInfo {
    /// Whether the Newton residual reached the configured tolerance.
    pub converged: bool,

    /// Number of Newton iterations performed.
    pub newton_iterations: usize,

    /// Final residual norm.
    pub final_residual: f64,

    /// Relative residual `||F|| / ||F0||`.
    pub relative_residual: f64,

    /// Total wall-clock time spent in the coupled step.
    pub wall_time_seconds: f64,

    /// Average GMRES iterations per Newton step.
    pub avg_gmres_iterations: usize,
}
