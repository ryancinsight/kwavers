//! Configuration for Discontinuous Galerkin solver

use crate::solver::spectral_dg::{BasisType, FluxType, LimiterType};

/// Configuration for DG solver
#[derive(Debug, Clone)]
pub struct DGConfig {
    /// Polynomial order (N means polynomials up to degree N)
    pub polynomial_order: usize,
    /// Basis function type
    pub basis_type: BasisType,
    /// Numerical flux type
    pub flux_type: FluxType,
    /// Enable slope limiting for shock capturing
    pub use_limiter: bool,
    /// Limiter type (if enabled)
    pub limiter_type: LimiterType,
}

impl Default for DGConfig {
    fn default() -> Self {
        Self {
            polynomial_order: 3,
            basis_type: BasisType::Legendre,
            flux_type: FluxType::LaxFriedrichs,
            use_limiter: true,
            limiter_type: LimiterType::Minmod,
        }
    }
}

impl DGConfig {
    /// Create a high-order configuration
    pub fn high_order(order: usize) -> Self {
        Self {
            polynomial_order: order,
            basis_type: BasisType::Legendre,
            flux_type: FluxType::Roe,
            use_limiter: false,
            limiter_type: LimiterType::None,
        }
    }
    
    /// Create a shock-capturing configuration
    pub fn shock_capturing(order: usize) -> Self {
        Self {
            polynomial_order: order,
            basis_type: BasisType::Legendre,
            flux_type: FluxType::LaxFriedrichs,
            use_limiter: true,
            limiter_type: LimiterType::WENO,
        }
    }
}