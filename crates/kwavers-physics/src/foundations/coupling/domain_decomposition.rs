//! Domain Decomposition for Multi-Physics Problems

use super::CouplingInterfaceCondition;
use aequitas::{
    dimension::DivideDimension,
    quantity::Quantity,
    systems::si::{
        dimensions,
        quantities::{Dimensionless, Length, ReciprocalLength, Time},
    },
};

/// Domain decomposition for multi-physics problems
#[derive(Debug, Clone)]
pub struct DomainDecomposition<T, D>
where
    D: DivideDimension<dimensions::Length>,
{
    /// Subdomain boundaries
    pub subdomain_bounds: Vec<Vec<Length<T>>>,
    /// Interface conditions between subdomains
    pub interface_conditions: Vec<CouplingInterfaceCondition>,
    /// Overlap region thickness (for overlapping Schwarz methods)
    pub overlap_thickness: Length<T>,
    /// Transmission conditions for domain coupling
    pub transmission_conditions: Vec<DomainDecompTransmissionCondition<T, D>>,
}

/// Transmission condition for domain decomposition
#[derive(Debug, Clone)]
pub enum DomainDecompTransmissionCondition<T, D>
where
    D: DivideDimension<dimensions::Length>,
{
    /// Dirichlet transmission: u = g
    Dirichlet {
        /// Prescribed interface field value `g`.
        boundary_value: Quantity<T, D>,
    },
    /// Neumann transmission: ∂u/∂n = g
    Neumann {
        /// Prescribed derivative normal to the interface.
        boundary_derivative: Quantity<T, <D as DivideDimension<dimensions::Length>>::Output>,
    },
    /// Robin transmission: αu + β∂u/∂n = g
    Robin {
        /// Coefficient α on the field value.
        alpha: Dimensionless<T>,
        /// Coefficient β on the normal derivative.
        beta: Length<T>,
        /// Right-hand-side interface value `g`.
        boundary_value: Quantity<T, D>,
    },
    /// Optimized Schwarz with optimized interface conditions
    OptimizedSchwarz {
        /// Optimized Schwarz interface-condition tuning parameter.
        optimization_parameter: ReciprocalLength<T>,
    },
}

/// Schwarz iteration method for domain decomposition
pub trait SchwarzMethod {
    /// Perform one Schwarz iteration
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn schwarz_iteration(&mut self, dt: Time) -> Result<(), String>;

    /// Check convergence of Schwarz method
    fn check_convergence(&self, tolerance: f64) -> bool;

    /// Get current iteration residual
    fn iteration_residual(&self) -> f64;
}
