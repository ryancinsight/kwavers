/// Statistics from iterative Born solution
#[derive(Debug, Clone)]
pub struct IterativeBornStats {
    /// Number of iterations performed
    pub iterations: usize,
    /// Final residual value
    pub final_residual: f64,
    /// History of residual values
    pub residual_history: Vec<f64>,
    /// Whether convergence was achieved
    pub converged: bool,
}

impl Default for IterativeBornStats {
    fn default() -> Self {
        Self {
            iterations: 0,
            final_residual: 0.0,
            residual_history: Vec::new(),
            converged: false,
        }
    }
}
