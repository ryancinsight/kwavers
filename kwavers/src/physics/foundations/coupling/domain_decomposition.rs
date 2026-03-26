//! Domain Decomposition for Multi-Physics Problems

use super::InterfaceCondition;

/// Domain decomposition for multi-physics problems
#[derive(Debug, Clone)]
pub struct DomainDecomposition {
    /// Subdomain boundaries
    pub subdomain_bounds: Vec<Vec<f64>>,
    /// Interface conditions between subdomains
    pub interface_conditions: Vec<InterfaceCondition>,
    /// Overlap region thickness (for overlapping Schwarz methods)
    pub overlap_thickness: f64,
    /// Transmission conditions for domain coupling
    pub transmission_conditions: Vec<TransmissionCondition>,
}

/// Transmission condition for domain decomposition
#[derive(Debug, Clone)]
pub enum TransmissionCondition {
    /// Dirichlet transmission: u = g
    Dirichlet { boundary_value: f64 },
    /// Neumann transmission: ∂u/∂n = g
    Neumann { boundary_flux: f64 },
    /// Robin transmission: αu + β∂u/∂n = g
    Robin {
        alpha: f64,
        beta: f64,
        boundary_value: f64,
    },
    /// Optimized Schwarz with optimized interface conditions
    OptimizedSchwarz { optimization_parameter: f64 },
}

/// Schwarz iteration method for domain decomposition
pub trait SchwarzMethod {
    /// Perform one Schwarz iteration
    fn schwarz_iteration(&mut self, dt: f64) -> Result<(), String>;

    /// Check convergence of Schwarz method
    fn check_convergence(&self, tolerance: f64) -> bool;

    /// Get current iteration residual
    fn iteration_residual(&self) -> f64;
}
