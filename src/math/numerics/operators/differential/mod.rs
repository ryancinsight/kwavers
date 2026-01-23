//! # Differential Operators
//!
//! This module provides finite difference operators for computing spatial derivatives.
//! All finite difference stencils in kwavers should be implemented here to ensure
//! consistency and avoid duplication.
//!
//! ## Operator Types
//!
//! - **Central Difference**: Standard central difference schemes (2nd, 4th, 6th order)
//! - **Staggered Grid**: Yee-style staggered grid operators for FDTD
//!
//! ## Mathematical Foundation
//!
//! For a function u(x), the first derivative is approximated by:
//!
//! **Second-order central:**
//! ```text
//! du/dx ≈ (u[i+1] - u[i-1]) / (2Δx) + O(Δx²)
//! ```
//!
//! **Fourth-order central:**
//! ```text
//! du/dx ≈ (-u[i+2] + 8u[i+1] - 8u[i-1] + u[i-2]) / (12Δx) + O(Δx⁴)
//! ```
//!
//! **Sixth-order central:**
//! ```text
//! du/dx ≈ (-u[i+3] + 9u[i+2] - 45u[i+1] + 45u[i-1] - 9u[i-2] + u[i-3]) / (60Δx) + O(Δx⁶)
//! ```
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use kwavers::math::numerics::operators::differential::{DifferentialOperator, CentralDifference2};
//! use ndarray::Array3;
//!
//! let op = CentralDifference2::new(0.001, 0.001, 0.001)?;
//! let field = Array3::zeros((100, 100, 100));
//! let gradient_x = op.apply_x(field.view())?;
//! ```
//!
//! ## Architecture
//!
//! The module follows a domain-driven design with clear separation of concerns:
//!
//! - `DifferentialOperator` trait: Algebraic interface defining differentiation contract
//! - Implementation modules: Each operator type in its own focused file
//! - Integration tests: Comprehensive validation against analytical solutions
//!
//! ## References
//!
//! - Fornberg, B. (1988). "Generation of finite difference formulas on arbitrarily
//!   spaced grids." *Mathematics of Computation*, 51(184), 699-706.
//!   DOI: 10.1090/S0025-5718-1988-0935077-0
//!
//! - Shubin, G. R., & Bell, J. B. (1987). "A modified equation approach to
//!   constructing fourth order methods for acoustic wave propagation."
//!   *SIAM Journal on Scientific and Statistical Computing*, 8(2), 135-151.
//!   DOI: 10.1137/0908025
//!
//! - Yee, K. (1966). "Numerical solution of initial boundary value problems
//!   involving Maxwell's equations in isotropic media."
//!   *IEEE Transactions on Antennas and Propagation*, 14(3), 302-307.
//!   DOI: 10.1109/TAP.1966.1138693

use crate::core::error::KwaversResult;
use ndarray::{Array3, ArrayView3};

// Implementation modules
mod central_difference_2;
mod central_difference_4;
mod central_difference_6;
mod staggered_grid;

// Re-export implementations
pub use central_difference_2::CentralDifference2;
pub use central_difference_4::CentralDifference4;
pub use central_difference_6::CentralDifference6;
pub use staggered_grid::StaggeredGridOperator;

// Integration tests
#[cfg(test)]
mod tests;

/// Trait for differential operators
///
/// This trait defines the algebraic interface for all spatial differentiation operators.
/// Implementations must provide methods for computing derivatives in each
/// spatial direction (X, Y, Z).
///
/// # Mathematical Properties
///
/// - **Order**: Accuracy order of the finite difference approximation
/// TODO_AUDIT: P2 - Advanced Numerical Methods - Implement high-order finite difference and spectral methods for PDE solving
/// DEPENDS ON: math/numerics/spectral_methods.rs, math/numerics/finite_elements.rs, math/numerics/weno_schemes.rs
/// MISSING: Weighted essentially non-oscillatory (WENO) schemes for shock capturing
/// MISSING: Discontinuous Galerkin finite element methods
/// MISSING: Spectral element methods with Gauss-Lobatto quadrature
/// MISSING: Compact finite difference schemes for reduced dispersion
/// MISSING: Arbitrary high-order Runge-Kutta methods for time integration
/// MISSING: Adaptive mesh refinement with solution-based error estimation
/// THEOREM: Lax equivalence theorem: Consistent + stable = convergent for well-posed PDEs
/// THEOREM: Godunov theorem: Linear monotone schemes cannot exceed first-order accuracy
/// THEOREM: Dahlquist barriers: A-stable linear multistep methods limited to order 2
/// REFERENCES: LeVeque (2002) Finite Volume Methods; Hesthaven (2007) Spectral Methods
/// - **Stencil Width**: Number of grid points used in the stencil
/// - **Conservation**: Whether the operator preserves conservation laws
/// - **Adjoint Consistency**: Whether the operator is consistent with its adjoint
///
/// # Thread Safety
///
/// All implementations must be `Send + Sync` to enable parallel computation.
///
/// # Contract
///
/// Implementations must satisfy:
///
/// 1. **Consistency**: As grid spacing → 0, operator → exact derivative
/// 2. **Stability**: Operator does not amplify numerical errors unboundedly
/// 3. **Conservation**: For conservative schemes, discrete conservation laws hold
/// 4. **Adjoint Consistency**: ⟨Ax, y⟩ = ⟨x, A*y⟩ for adjoint-consistent operators
pub trait DifferentialOperator: Send + Sync {
    /// Apply operator to compute ∂u/∂x
    ///
    /// # Arguments
    ///
    /// * `field` - Input field u(x,y,z)
    ///
    /// # Returns
    ///
    /// Derivative ∂u/∂x with same dimensions as input
    ///
    /// # Errors
    ///
    /// Returns error if field dimensions are insufficient for stencil width
    ///
    /// # Mathematical Specification
    ///
    /// For a smooth function u(x,y,z), computes:
    /// ```text
    /// ∂u/∂x + O(Δx^p)
    /// ```
    /// where p is the order of accuracy.
    fn apply_x(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>>;

    /// Apply operator to compute ∂u/∂y
    ///
    /// # Arguments
    ///
    /// * `field` - Input field u(x,y,z)
    ///
    /// # Returns
    ///
    /// Derivative ∂u/∂y with same dimensions as input
    ///
    /// # Errors
    ///
    /// Returns error if field dimensions are insufficient for stencil width
    fn apply_y(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>>;

    /// Apply operator to compute ∂u/∂z
    ///
    /// # Arguments
    ///
    /// * `field` - Input field u(x,y,z)
    ///
    /// # Returns
    ///
    /// Derivative ∂u/∂z with same dimensions as input
    ///
    /// # Errors
    ///
    /// Returns error if field dimensions are insufficient for stencil width
    fn apply_z(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>>;

    /// Get the accuracy order of the operator
    ///
    /// # Returns
    ///
    /// Order of accuracy (e.g., 2 for O(Δx²), 4 for O(Δx⁴))
    ///
    /// # Mathematical Specification
    ///
    /// If the exact derivative is D[u] and the numerical approximation is D_h[u],
    /// then:
    /// ```text
    /// |D[u] - D_h[u]| = O(h^p)
    /// ```
    /// where p is the returned order.
    fn order(&self) -> usize;

    /// Get the stencil width
    ///
    /// # Returns
    ///
    /// Number of grid points used in the stencil
    ///
    /// # Usage
    ///
    /// This is used to validate that input grids have sufficient points
    /// to apply the operator without out-of-bounds access.
    fn stencil_width(&self) -> usize;

    /// Check if operator is conservative
    ///
    /// Conservative operators preserve discrete conservation laws.
    ///
    /// # Mathematical Property
    ///
    /// A conservative operator satisfies:
    /// ```text
    /// ∑_i D[u_i] = 0
    /// ```
    /// for appropriate boundary conditions.
    ///
    /// # Default
    ///
    /// Returns `false` unless overridden by implementation.
    fn is_conservative(&self) -> bool {
        false
    }

    /// Check if operator is adjoint-consistent
    ///
    /// Adjoint-consistent operators satisfy: ⟨Ax, y⟩ = ⟨x, A*y⟩
    ///
    /// # Mathematical Property
    ///
    /// For inner product ⟨·,·⟩ and adjoint operator A*, the operator A
    /// is adjoint-consistent if:
    /// ```text
    /// ⟨Ax, y⟩ = ⟨x, A*y⟩
    /// ```
    /// for all appropriate x, y.
    ///
    /// # Importance
    ///
    /// Adjoint consistency is critical for:
    /// - Optimization problems (gradient accuracy)
    /// - Inverse problems (adjoint-state methods)
    /// - Stability of coupled systems
    ///
    /// # Default
    ///
    /// Returns `true` (most standard FD stencils are adjoint-consistent).
    fn is_adjoint_consistent(&self) -> bool {
        true
    }
}
