//! Configuration for Discontinuous Galerkin solver

use super::basis::BasisType;
use super::flux::{FluxType, LimiterType};

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
    /// Create configuration for high-order accurate simulations
    pub fn high_order() -> Self {
        Self {
            polynomial_order: 5,
            basis_type: BasisType::Legendre,
            flux_type: FluxType::Roe,
            use_limiter: false,
            limiter_type: LimiterType::None,
        }
    }

    /// Create configuration for shock-capturing simulations
    pub fn shock_capturing() -> Self {
        Self {
            polynomial_order: 2,
            basis_type: BasisType::Legendre,
            flux_type: FluxType::LaxFriedrichs,
            use_limiter: true,
            limiter_type: LimiterType::VanAlbada,
        }
    }
}
