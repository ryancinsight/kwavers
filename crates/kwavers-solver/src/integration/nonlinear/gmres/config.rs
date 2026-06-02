//! GMRES configuration parameters.

/// GMRES solver configuration.
#[derive(Debug, Clone)]
pub struct GMRESConfig {
    /// Krylov subspace dimension before restart (typical: 10–100).
    pub krylov_dim: usize,
    /// Maximum number of outer restart iterations.
    pub max_iterations: usize,
    /// Relative tolerance: ||r|| / ||b|| < tol.
    pub relative_tolerance: f64,
    /// Absolute tolerance: ||r|| < tol.
    pub absolute_tolerance: f64,
    /// Enable preconditioning.
    pub use_preconditioner: bool,
}

impl Default for GMRESConfig {
    fn default() -> Self {
        Self {
            krylov_dim: 30,
            max_iterations: 100,
            relative_tolerance: 1e-6,
            absolute_tolerance: 1e-10,
            use_preconditioner: false,
        }
    }
}
