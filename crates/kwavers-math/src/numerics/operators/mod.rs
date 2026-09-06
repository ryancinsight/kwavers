//! # Numerical Operators Module
//!
//! Spectral operations, interpolation, and the summation-by-parts family used
//! by the kwavers solvers.
//!
//! ## Where the first-derivative stencils are
//!
//! Not here. Central differences, the Yee staggered pair, and the
//! arbitrary-even-order staggered gradient/divergence pair are Leto's
//! ([`leto_ops::FiniteDifference3D`], [`leto_ops::StaggeredLeapfrog3D`]), so
//! one implementation serves every consumer in the stack and a new order is a
//! parameter rather than another cloned kernel. This module keeps only what
//! Leto does not own.
//!
//! ## Architecture
//!
//! The operators module defines two core trait families:
//!
//! - **`SpectralOperatorTrait`**: FFT-based operations in k-space
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
//! - `differential`: summation-by-parts operators with their energy norms
//! - `spectral`: Pseudospectral operators using FFT
//! - `interpolation`: Spatial interpolation (linear, cubic, conservative)
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use leto::Array3;
//! use leto_ops::{Axis, StaggeredLeapfrog3D};
//!
//! let dx = 0.001; // 1 mm grid spacing
//! let op = StaggeredLeapfrog3D::new(4, dx, dx, dx)?;
//!
//! let field = Array3::zeros([100, 100, 100]);
//! let mut gradient_x = Array3::zeros([100, 100, 100]);
//! op.gradient_into(Axis::X, field.view(), &mut gradient_x.view_mut())?;
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
pub use interpolation::Interpolator;
pub use spectral::SpectralOperatorTrait;

// Re-export common implementations
pub use differential::summation_by_parts::SummationByPartsOperator;
pub use interpolation::{LinearInterpolator, NumericsTrilinearInterpolator};
pub use spectral::PseudospectralDerivative;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_traits_are_object_safe() {
        // Verify traits can be used as trait objects if needed
        // Note: This is a compile-time check
        fn _assert_spectral_object_safe(_: &dyn SpectralOperatorTrait) {}
        fn _assert_interpolator_object_safe(_: &dyn Interpolator) {}
    }
}
