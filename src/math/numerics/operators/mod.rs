//! # Numerical Operators Module
//!
//! This module provides trait definitions and implementations for all numerical
//! operators used in kwavers. All spatial derivatives, spectral operations, and
//! interpolation should use these unified interfaces.
//!
//! ## Architecture
//!
//! The operators module defines three core trait families:
//!
//! - **`DifferentialOperator`**: Finite difference and spectral differentiation
//! - **`SpectralOperator`**: FFT-based operations in k-space
//! - **`Interpolator`**: Spatial interpolation for heterogeneous media
//!
//! ## Design Principles
//!
//! 1. **Trait-Based Polymorphism**: All operators implement common traits
//! 2. **Compile-Time Dispatch**: Zero-cost abstractions via monomorphization
//! 3. **Conservation Properties**: Operators preserve physical invariants
//! 4. **Adjoint Consistency**: Support for adjoint-based methods
//!
//! ## Modules
//!
//! - `differential`: Finite difference stencils (central, upwind, staggered)
//! - `spectral`: Pseudospectral operators using FFT
//! - `interpolation`: Spatial interpolation (linear, cubic, conservative)
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use kwavers::math::numerics::operators::{DifferentialOperator, CentralDifference2};
//! use ndarray::Array3;
//!
//! let dx = 0.001; // 1 mm grid spacing
//! let op = CentralDifference2::new(dx, dx, dx)?;
//!
//! let field = Array3::zeros((100, 100, 100));
//! let gradient_x = op.apply_x(field.view())?;
//! ```
//!
//! ## References
//!
//! - Fornberg, B. (1988). "Generation of finite difference formulas on arbitrarily
//!   spaced grids." *Mathematics of Computation*, 51(184), 699-706.
//! - Shubin, G. R., & Bell, J. B. (1987). "A modified equation approach to
//!   constructing fourth order methods for acoustic wave propagation."
//!   *SIAM Journal on Scientific and Statistical Computing*, 8(2), 135-151.

pub mod differential;
pub mod interpolation;
pub mod spectral;

// Re-export main traits for convenience
pub use differential::DifferentialOperator;
pub use interpolation::Interpolator;
pub use spectral::SpectralOperator;

// Re-export common implementations
pub use differential::{
    CentralDifference2, CentralDifference4, CentralDifference6, StaggeredGridOperator,
};
pub use interpolation::{LinearInterpolator, TrilinearInterpolator};
pub use spectral::PseudospectralDerivative;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_traits_are_object_safe() {
        // Verify traits can be used as trait objects if needed
        // Note: This is a compile-time check
        fn _assert_differential_object_safe(_: &dyn DifferentialOperator) {}
        fn _assert_spectral_object_safe(_: &dyn SpectralOperator) {}
        fn _assert_interpolator_object_safe(_: &dyn Interpolator) {}
    }
}
