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
//! - **`FiniteDifference3D` / `FiniteDifference3DScheme`**: provider-SSOT
//!   finite-difference stencils in `leto-ops`, re-exported here for convenience.
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
//! - `differential`: Finite difference stencils (central, upwind, staggered)
//! - `spectral`: Pseudospectral operators using FFT
//! - `interpolation`: Spatial interpolation (linear, cubic, conservative)
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use leto::Array3;
//! // SSOT shim pattern: kwavers callers import through the kwavers-side
//! // re-export rather than reaching into leto-ops directly. The re-export
//! // resolves to the same leto types (ADR 0018 / ADR 0033). The names
//! // `FiniteDifference3D` and `FiniteDifference3DScheme` are in scope here
//! // because of the `pub use differential::{...}` line below.
//! let dx = 0.001; // 1 mm grid spacing
//!
//! // Provider-SSOT (ADR 0018 / ADR 0033): all 3-D finite-difference kernels
//! // live in leto-ops. The kwavers-side differential module re-exports the
//! // same types so callers can keep using
//! // `kwavers_math::numerics::operators::FiniteDifference3D`.
//! let op = match FiniteDifference3D::new(
//!     FiniteDifference3DScheme::CentralSecondOrder,
//!     dx, dx, dx,
//! ) {
//!     Ok(op) => op,
//!     Err(e) => panic!("Leto-level construction failure: {e}"),
//! };
//!
//! let field = Array3::<f64>::zeros([100, 100, 100]);
//! let mut gradient_x = Array3::<f64>::zeros([100, 100, 100]);
//! op.apply_x_into(field.view(), &mut gradient_x);
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
pub use differential::{FiniteDifference3D, FiniteDifference3DScheme};
pub use interpolation::Interpolator;
pub use spectral::SpectralOperatorTrait;

// Re-export common implementations
// All 3-D finite-difference kernels are provider-SSOT in leto-ops
// (ADR 0018 / ADR 0033).
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
