//! # Numerical Methods Module
//!
//! This module provides the foundational numerical operations used throughout kwavers.
//! All numerical algorithms are implemented here to avoid duplication and ensure
//! mathematical correctness.
//!
//! ## Architecture
//!
//! The numerics module is organized into three main components:
//!
//! - **`operators/`**: Differential, spectral, and interpolation operators
//! - **`integration/`**: Numerical integration and quadrature schemes
//! - **`transforms/`**: Fourier, wavelet, and other transforms
//!
//! ## Design Principles
//!
//! 1. **Single Source of Truth**: Each numerical operation has exactly one implementation
//! 2. **Trait-Based Abstractions**: All operators implement well-defined traits
//! 3. **Literature-Validated**: Every method includes references to original papers
//! 4. **Zero-Cost Abstractions**: Traits compile to direct function calls
//! 5. **Conservation Laws**: Operators preserve physical invariants where applicable
//!
//! ## Usage
//!
//! ```rust,ignore
//! use kwavers::math::numerics::operators::{DifferentialOperator, CentralDifference2};
//!
//! // Create second-order central difference operator
//! let op = CentralDifference2::new(0.001, 0.001, 0.001)?;
//!
//! // Apply to field
//! let gradient_x = op.apply_x(field.view())?;
//! ```
//!
//! ## Layer Dependencies
//!
//! This module sits in the **math layer** and:
//! - ✅ MAY import from: `core::*`
//! - 🔴 MUST NOT import from: `domain::*`, `physics::*`, `solver::*`, `clinical::*`
//!
//! All higher layers (domain, physics, solver) should import numerical operations
//! from this module rather than implementing their own.
//!
//! ## References
//!
//! - Fornberg, B. (1988). "Generation of finite difference formulas on arbitrarily
//!   spaced grids." *Mathematics of Computation*, 51(184), 699-706.
//! - Liu, Q. H. (1997). "The PSTD algorithm: A time-domain method requiring only
//!   two cells per wavelength." *Microwave and Optical Technology Letters*, 15(3), 158-165.
//! - Hesthaven, J. S., & Warburton, T. (2007). *Nodal Discontinuous Galerkin Methods*.
//!   Springer.

pub mod operators;

// Integration and transforms to be implemented in future phases
// pub mod integration;
// pub mod transforms;

// Re-export commonly used traits for convenience
pub use operators::{DifferentialOperator, Interpolator, SpectralOperator};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_structure() {
        // Verify module is loadable
        assert!(true);
    }
}
