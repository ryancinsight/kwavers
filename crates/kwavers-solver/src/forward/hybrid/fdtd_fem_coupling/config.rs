/// Configuration for FDTD-FEM coupling
#[derive(Debug, Clone)]
pub struct FdtdFemCouplingConfig {
    /// Relaxation parameter for Schwarz method (0 < omega <= 1)
    pub relaxation_factor: f64,
    /// Maximum iterations for convergence
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Interface thickness for smoothing
    pub interface_thickness: f64,
}

impl Default for FdtdFemCouplingConfig {
    fn default() -> Self {
        Self {
            relaxation_factor: 0.8, // Conservative relaxation
            max_iterations: 10,
            tolerance: 1e-6,
            interface_thickness: 2.0, // 2 grid cells
        }
    }
}
