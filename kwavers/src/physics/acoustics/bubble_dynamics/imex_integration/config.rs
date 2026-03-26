//! Configuration for IMEX bubble integration

/// Configuration for IMEX bubble integration
#[derive(Debug, Clone)]
pub struct BubbleIMEXConfig {
    /// Relative tolerance for implicit solver
    pub rtol: f64,
    /// Absolute tolerance for implicit solver
    pub atol: f64,
    /// Maximum iterations for implicit solver
    pub max_iter: usize,
    /// Enable adaptive time stepping
    pub adaptive: bool,
    /// Minimum time step for adaptive stepping
    pub dt_min: f64,
    /// Maximum time step for adaptive stepping
    pub dt_max: f64,
}

impl Default for BubbleIMEXConfig {
    fn default() -> Self {
        Self {
            rtol: 1e-6,
            atol: 1e-9,
            max_iter: 10,
            adaptive: false,
            dt_min: 1e-12,
            dt_max: 1e-7,
        }
    }
}
