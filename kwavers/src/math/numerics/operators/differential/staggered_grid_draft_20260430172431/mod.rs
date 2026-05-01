//! # Staggered Grid Operator
//!
//! This module implements the staggered grid finite difference operator,
//! commonly known as the Yee scheme in electromagnetics. The staggered grid
//! approach places different field components at offset locations, enabling
//! natural conservation properties.
//!
//! ## Mathematical Specification
//!
//! For a staggered grid, field components are offset by half a grid cell:
//!
//! **Forward difference (pressure → velocity):**
//! ```text
//! du/dx|_{i+1/2} ≈ (u[i+1] - u[i]) / Δx + O(Δx²)
//! ```
//!
//! **Backward difference (velocity → pressure):**
//! ```text
//! du/dx|_i ≈ (u[i] - u[i-1]) / Δx + O(Δx²)
//! ```
//!
//! ## Stencil
//!
//! The operator uses a 2-point stencil:
//! ```text
//! Forward:  [i, i+1] with coefficients [-1/Δx, +1/Δx]
//! Backward: [i-1, i] with coefficients [-1/Δx, +1/Δx]
//! ```
//!
//! ## Properties
//!
//! - **Order**: 2 (second-order accurate on staggered grid)
//! - **Stencil Width**: 2 points
//! - **Conservation**: Yes (discrete conservation laws preserved)
//! - **Adjoint Consistency**: Yes (structure-preserving)
//!
//! ## Module layout
//!
//! - [`forward`]: forward-difference kernels (cell-center → cell-edge),
//!   zero-allocation `_into` plus allocating wrappers per axis.
//! - [`backward`]: backward-difference kernels (cell-edge → cell-center)
//!   with first-cell forward-difference fallback for the `i = 0` boundary.
//!
//! ## Applications
//!
//! Staggered grids are particularly well-suited for:
//! - FDTD electromagnetic simulations
//! - Acoustic wave propagation with velocity-stress formulation
//! - Incompressible fluid dynamics (MAC grids)
//!
//! ## References
//!
//! - Yee, K. (1966). "Numerical solution of initial boundary value problems
//!   involving Maxwell's equations in isotropic media."
//!   *IEEE Transactions on Antennas and Propagation*, 14(3), 302-307.
//!   DOI: 10.1109/TAP.1966.1138693
//!
//! - Virieux, J. (1986). "P-SV wave propagation in heterogeneous media:
//!   Velocity-stress finite-difference method."
//!   *Geophysics*, 51(4), 889-901.
//!   DOI: 10.1190/1.1442147

mod backward;
mod forward;

#[cfg(test)]
mod tests;

use ndarray::{Array3, ArrayView3};

use super::DifferentialOperator;
use crate::core::error::{KwaversResult, NumericalError};

/// Staggered grid finite difference operator
///
/// This operator implements the Yee-style staggered grid scheme where field
/// components are offset by half a grid cell. This staggering provides natural
/// conservation properties and second-order accuracy.
///
/// # Grid Layout
///
/// On a staggered grid, scalar fields (e.g., pressure) and vector fields
/// (e.g., velocity components) are positioned at different locations:
///
/// ```text
/// Pressure:  p[i]       p[i+1]     p[i+2]
///            |          |          |
/// Velocity:     u[i+1/2]   u[i+3/2]
/// ```
///
/// # Forward vs Backward Differences
///
/// - **Forward difference**: Used to compute derivatives at cell edges from
///   cell centers (e.g., ∂p/∂x at velocity points from pressure values)
/// - **Backward difference**: Used to compute derivatives at cell centers from
///   cell edges (e.g., ∂u/∂x at pressure points from velocity values)
///
/// # Conservation Properties
///
/// The staggered grid naturally preserves discrete conservation laws. For
/// example, in acoustics:
/// ```text
/// ∂p/∂t + ρc² ∇·u = 0
/// ∂u/∂t + (1/ρ) ∇p = 0
/// ```
/// The discrete analogs exactly conserve energy when using the staggered scheme.
#[derive(Debug, Clone)]
pub struct StaggeredGridOperator {
    /// Grid spacing in X direction (meters)
    pub(crate) dx: f64,
    /// Grid spacing in Y direction (meters)
    pub(crate) dy: f64,
    /// Grid spacing in Z direction (meters)
    pub(crate) dz: f64,
}

impl StaggeredGridOperator {
    /// Create a new staggered grid operator
    ///
    /// # Arguments
    ///
    /// * `dx` - Grid spacing in X direction (meters)
    /// * `dy` - Grid spacing in Y direction (meters)
    /// * `dz` - Grid spacing in Z direction (meters)
    ///
    /// # Errors
    ///
    /// Returns error if any grid spacing is non-positive.
    pub fn new(dx: f64, dy: f64, dz: f64) -> KwaversResult<Self> {
        if dx <= 0.0 || dy <= 0.0 || dz <= 0.0 {
            return Err(NumericalError::InvalidGridSpacing { dx, dy, dz }.into());
        }
        Ok(Self { dx, dy, dz })
    }
}

impl DifferentialOperator for StaggeredGridOperator {
    /// Apply operator in X direction (default: forward difference)
    ///
    /// For the `DifferentialOperator` trait implementation, defaults to
    /// forward difference. Use `apply_forward_x` or `apply_backward_x`
    /// directly for explicit control.
    fn apply_x(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        self.apply_forward_x(field)
    }

    /// Apply operator in Y direction (default: forward difference)
    fn apply_y(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        self.apply_forward_y(field)
    }

    /// Apply operator in Z direction (default: forward difference)
    fn apply_z(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        self.apply_forward_z(field)
    }

    fn order(&self) -> usize {
        2
    }

    fn stencil_width(&self) -> usize {
        2
    }

    fn is_conservative(&self) -> bool {
        true // Staggered grids preserve discrete conservation laws
    }
}
