//! Domain Decomposition for Multi-Physics Problems

use super::CouplingInterfaceCondition;

/// Domain decomposition for multi-physics problems
#[derive(Debug, Clone)]
pub struct DomainDecomposition {
    /// Subdomain boundaries
    pub subdomain_bounds: Vec<Vec<f64>>,
    /// Interface conditions between subdomains
    pub interface_conditions: Vec<CouplingInterfaceCondition>,
    /// Overlap region thickness (for overlapping Schwarz methods)
    pub overlap_thickness: f64,
    /// Transmission conditions for domain coupling
    pub transmission_conditions: Vec<DomainDecompTransmissionCondition>,
}

/// Transmission condition for domain decomposition
#[derive(Debug, Clone)]
pub enum DomainDecompTransmissionCondition {
    /// Dirichlet transmission: u = g
    Dirichlet {
        /// Prescribed interface field value `g`.
        boundary_value: f64,
    },
    /// Neumann transmission: ∂u/∂n = g
    Neumann {
        /// Prescribed interface normal flux `g = ∂u/∂n`.
        boundary_flux: f64,
    },
    /// Robin transmission: αu + β∂u/∂n = g
    Robin {
        /// Coefficient α on the field value.
        alpha: f64,
        /// Coefficient β on the normal derivative.
        beta: f64,
        /// Right-hand-side interface value `g`.
        boundary_value: f64,
    },
    /// Optimized Schwarz with optimized interface conditions
    OptimizedSchwarz {
        /// Optimized Schwarz interface-condition tuning parameter.
        optimization_parameter: f64,
    },
}

/// Schwarz iteration method for domain decomposition
pub trait SchwarzMethod {
    /// Perform one Schwarz iteration
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn schwarz_iteration(&mut self, dt: f64) -> Result<(), String>;

    /// Check convergence of Schwarz method
    fn check_convergence(&self, tolerance: f64) -> bool;

    /// Get current iteration residual
    fn iteration_residual(&self) -> f64;
}
