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
//! - **Upwind**: Upwind-biased schemes for convection-dominated problems
//! - **Conservative**: Conservative finite difference for shock-capturing
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
//! ## Usage Example
//!
//! ```rust,ignore
//! use kwavers::math::numerics::operators::{DifferentialOperator, CentralDifference2};
//! use ndarray::Array3;
//!
//! let op = CentralDifference2::new(0.001, 0.001, 0.001)?;
//! let field = Array3::zeros((100, 100, 100));
//! let gradient_x = op.apply_x(field.view())?;
//! ```
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

use crate::core::error::{KwaversResult, NumericalError};
use ndarray::{Array3, ArrayView3, Axis};

/// Trait for differential operators
///
/// This trait defines the interface for all spatial differentiation operators.
/// Implementations must provide methods for computing derivatives in each
/// spatial direction (X, Y, Z).
///
/// # Mathematical Properties
///
/// - **Order**: Accuracy order of the finite difference approximation
/// - **Stencil Width**: Number of grid points used in the stencil
/// - **Conservation**: Whether the operator preserves conservation laws
/// - **Adjoint Consistency**: Whether the operator is consistent with its adjoint
///
/// # Thread Safety
///
/// All implementations must be `Send + Sync` to enable parallel computation.
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
    fn apply_z(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>>;

    /// Get the accuracy order of the operator
    ///
    /// # Returns
    ///
    /// Order of accuracy (e.g., 2 for O(Δx²), 4 for O(Δx⁴))
    fn order(&self) -> usize;

    /// Get the stencil width
    ///
    /// # Returns
    ///
    /// Number of grid points used in the stencil
    fn stencil_width(&self) -> usize;

    /// Check if operator is conservative
    ///
    /// Conservative operators preserve discrete conservation laws
    fn is_conservative(&self) -> bool {
        false
    }

    /// Check if operator is adjoint-consistent
    ///
    /// Adjoint-consistent operators satisfy: ⟨Ax, y⟩ = ⟨x, A*y⟩
    fn is_adjoint_consistent(&self) -> bool {
        false
    }
}

/// Second-order accurate central difference operator
///
/// This operator uses a 3-point stencil to compute spatial derivatives with
/// O(Δx²) accuracy.
///
/// # Stencil
///
/// ```text
/// du/dx ≈ (u[i+1] - u[i-1]) / (2Δx)
/// ```
///
/// # Boundary Treatment
///
/// - Interior points: Central difference
/// - Boundary points: Forward/backward difference (1st order)
///
/// # References
///
/// Standard finite difference method, see any numerical analysis textbook.
#[derive(Debug, Clone)]
pub struct CentralDifference2 {
    /// Grid spacing in X direction
    dx: f64,
    /// Grid spacing in Y direction
    dy: f64,
    /// Grid spacing in Z direction
    dz: f64,
}

impl CentralDifference2 {
    /// Create a new second-order central difference operator
    ///
    /// # Arguments
    ///
    /// * `dx` - Grid spacing in X direction (meters)
    /// * `dy` - Grid spacing in Y direction (meters)
    /// * `dz` - Grid spacing in Z direction (meters)
    ///
    /// # Returns
    ///
    /// New operator instance
    ///
    /// # Errors
    ///
    /// Returns error if any grid spacing is non-positive
    pub fn new(dx: f64, dy: f64, dz: f64) -> KwaversResult<Self> {
        if dx <= 0.0 || dy <= 0.0 || dz <= 0.0 {
            return Err(NumericalError::InvalidGridSpacing { dx, dy, dz }.into());
        }
        Ok(Self { dx, dy, dz })
    }
}

impl DifferentialOperator for CentralDifference2 {
    fn apply_x(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();

        if nx < 3 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 3,
                actual: nx,
                direction: "X".to_string(),
            }
            .into());
        }

        let mut result = Array3::zeros((nx, ny, nz));

        // Interior points: central difference
        for i in 1..nx - 1 {
            for j in 0..ny {
                for k in 0..nz {
                    result[[i, j, k]] =
                        (field[[i + 1, j, k]] - field[[i - 1, j, k]]) / (2.0 * self.dx);
                }
            }
        }

        // Boundary points: forward/backward difference (1st order)
        for j in 0..ny {
            for k in 0..nz {
                // Left boundary: forward difference
                result[[0, j, k]] = (field[[1, j, k]] - field[[0, j, k]]) / self.dx;
                // Right boundary: backward difference
                result[[nx - 1, j, k]] = (field[[nx - 1, j, k]] - field[[nx - 2, j, k]]) / self.dx;
            }
        }

        Ok(result)
    }

    fn apply_y(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();

        if ny < 3 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 3,
                actual: ny,
                direction: "Y".to_string(),
            }
            .into());
        }

        let mut result = Array3::zeros((nx, ny, nz));

        // Interior points: central difference
        for i in 0..nx {
            for j in 1..ny - 1 {
                for k in 0..nz {
                    result[[i, j, k]] =
                        (field[[i, j + 1, k]] - field[[i, j - 1, k]]) / (2.0 * self.dy);
                }
            }
        }

        // Boundary points: forward/backward difference
        for i in 0..nx {
            for k in 0..nz {
                result[[i, 0, k]] = (field[[i, 1, k]] - field[[i, 0, k]]) / self.dy;
                result[[i, ny - 1, k]] = (field[[i, ny - 1, k]] - field[[i, ny - 2, k]]) / self.dy;
            }
        }

        Ok(result)
    }

    fn apply_z(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();

        if nz < 3 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 3,
                actual: nz,
                direction: "Z".to_string(),
            }
            .into());
        }

        let mut result = Array3::zeros((nx, ny, nz));

        // Interior points: central difference
        for i in 0..nx {
            for j in 0..ny {
                for k in 1..nz - 1 {
                    result[[i, j, k]] =
                        (field[[i, j, k + 1]] - field[[i, j, k - 1]]) / (2.0 * self.dz);
                }
            }
        }

        // Boundary points: forward/backward difference
        for i in 0..nx {
            for j in 0..ny {
                result[[i, j, 0]] = (field[[i, j, 1]] - field[[i, j, 0]]) / self.dz;
                result[[i, j, nz - 1]] = (field[[i, j, nz - 1]] - field[[i, j, nz - 2]]) / self.dz;
            }
        }

        Ok(result)
    }

    fn order(&self) -> usize {
        2
    }

    fn stencil_width(&self) -> usize {
        3
    }

    fn is_adjoint_consistent(&self) -> bool {
        true // Symmetric stencil
    }
}

/// Fourth-order accurate central difference operator
///
/// This operator uses a 5-point stencil to compute spatial derivatives with
/// O(Δx⁴) accuracy.
///
/// # Stencil (Fornberg 1988)
///
/// ```text
/// du/dx ≈ (-u[i+2] + 8u[i+1] - 8u[i-1] + u[i-2]) / (12Δx)
/// ```
///
/// # Boundary Treatment
///
/// - Interior points (i ≥ 2, i < n-2): Fourth-order central
/// - Near-boundary (i=1, i=n-2): Second-order central fallback
/// - Boundary points (i=0, i=n-1): First-order forward/backward
///
/// # References
///
/// - Fornberg, B. (1988). Mathematics of Computation, 51(184), 699-706.
#[derive(Debug, Clone)]
pub struct CentralDifference4 {
    dx: f64,
    dy: f64,
    dz: f64,
}

impl CentralDifference4 {
    /// Create a new fourth-order central difference operator
    pub fn new(dx: f64, dy: f64, dz: f64) -> KwaversResult<Self> {
        if dx <= 0.0 || dy <= 0.0 || dz <= 0.0 {
            return Err(NumericalError::InvalidGridSpacing { dx, dy, dz }.into());
        }
        Ok(Self { dx, dy, dz })
    }
}

impl DifferentialOperator for CentralDifference4 {
    fn apply_x(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();

        if nx < 5 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 5,
                actual: nx,
                direction: "X".to_string(),
            }
            .into());
        }

        let mut result = Array3::zeros((nx, ny, nz));

        // Interior points: fourth-order central
        for i in 2..nx - 2 {
            for j in 0..ny {
                for k in 0..nz {
                    result[[i, j, k]] = (-field[[i + 2, j, k]] + 8.0 * field[[i + 1, j, k]]
                        - 8.0 * field[[i - 1, j, k]]
                        + field[[i - 2, j, k]])
                        / (12.0 * self.dx);
                }
            }
        }

        // Near-boundary: second-order central
        for j in 0..ny {
            for k in 0..nz {
                if nx > 2 {
                    result[[1, j, k]] = (field[[2, j, k]] - field[[0, j, k]]) / (2.0 * self.dx);
                    result[[nx - 2, j, k]] =
                        (field[[nx - 1, j, k]] - field[[nx - 3, j, k]]) / (2.0 * self.dx);
                }
            }
        }

        // Boundary points: first-order
        for j in 0..ny {
            for k in 0..nz {
                result[[0, j, k]] = (field[[1, j, k]] - field[[0, j, k]]) / self.dx;
                result[[nx - 1, j, k]] = (field[[nx - 1, j, k]] - field[[nx - 2, j, k]]) / self.dx;
            }
        }

        Ok(result)
    }

    fn apply_y(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();

        if ny < 5 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 5,
                actual: ny,
                direction: "Y".to_string(),
            }
            .into());
        }

        let mut result = Array3::zeros((nx, ny, nz));

        // Interior points: fourth-order central
        for i in 0..nx {
            for j in 2..ny - 2 {
                for k in 0..nz {
                    result[[i, j, k]] = (-field[[i, j + 2, k]] + 8.0 * field[[i, j + 1, k]]
                        - 8.0 * field[[i, j - 1, k]]
                        + field[[i, j - 2, k]])
                        / (12.0 * self.dy);
                }
            }
        }

        // Near-boundary and boundary points (similar to apply_x)
        for i in 0..nx {
            for k in 0..nz {
                if ny > 2 {
                    result[[i, 1, k]] = (field[[i, 2, k]] - field[[i, 0, k]]) / (2.0 * self.dy);
                    result[[i, ny - 2, k]] =
                        (field[[i, ny - 1, k]] - field[[i, ny - 3, k]]) / (2.0 * self.dy);
                }
                result[[i, 0, k]] = (field[[i, 1, k]] - field[[i, 0, k]]) / self.dy;
                result[[i, ny - 1, k]] = (field[[i, ny - 1, k]] - field[[i, ny - 2, k]]) / self.dy;
            }
        }

        Ok(result)
    }

    fn apply_z(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();

        if nz < 5 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 5,
                actual: nz,
                direction: "Z".to_string(),
            }
            .into());
        }

        let mut result = Array3::zeros((nx, ny, nz));

        // Interior points: fourth-order central
        for i in 0..nx {
            for j in 0..ny {
                for k in 2..nz - 2 {
                    result[[i, j, k]] = (-field[[i, j, k + 2]] + 8.0 * field[[i, j, k + 1]]
                        - 8.0 * field[[i, j, k - 1]]
                        + field[[i, j, k - 2]])
                        / (12.0 * self.dz);
                }
            }
        }

        // Near-boundary and boundary points
        for i in 0..nx {
            for j in 0..ny {
                if nz > 2 {
                    result[[i, j, 1]] = (field[[i, j, 2]] - field[[i, j, 0]]) / (2.0 * self.dz);
                    result[[i, j, nz - 2]] =
                        (field[[i, j, nz - 1]] - field[[i, j, nz - 3]]) / (2.0 * self.dz);
                }
                result[[i, j, 0]] = (field[[i, j, 1]] - field[[i, j, 0]]) / self.dz;
                result[[i, j, nz - 1]] = (field[[i, j, nz - 1]] - field[[i, j, nz - 2]]) / self.dz;
            }
        }

        Ok(result)
    }

    fn order(&self) -> usize {
        4
    }

    fn stencil_width(&self) -> usize {
        5
    }

    fn is_adjoint_consistent(&self) -> bool {
        true // Symmetric stencil
    }
}

/// Staggered grid operator for FDTD simulations
///
/// This operator implements the Yee scheme for staggered grids, where field
/// components are offset by half a grid cell. This is the standard approach
/// in FDTD methods for electromagnetics and acoustics.
///
/// # Grid Layout
///
/// ```text
/// Pressure p: grid points at integer indices [i, j, k]
/// Velocity v: grid points at half-integer indices [i+1/2, j+1/2, k+1/2]
/// ```
///
/// # Mathematical Form
///
/// For forward difference (pressure to velocity):
/// ```text
/// dp/dx ≈ (p[i+1] - p[i]) / Δx
/// ```
///
/// For backward difference (velocity to pressure):
/// ```text
/// dv/dx ≈ (v[i] - v[i-1]) / Δx
/// ```
///
/// # References
///
/// - Yee, K. (1966). IEEE Trans. Antennas Propag., 14(3), 302-307.
#[derive(Debug, Clone)]
pub struct StaggeredGridOperator {
    dx: f64,
    dy: f64,
    dz: f64,
}

impl StaggeredGridOperator {
    /// Create a new staggered grid operator
    pub fn new(dx: f64, dy: f64, dz: f64) -> KwaversResult<Self> {
        if dx <= 0.0 || dy <= 0.0 || dz <= 0.0 {
            return Err(NumericalError::InvalidGridSpacing { dx, dy, dz }.into());
        }
        Ok(Self { dx, dy, dz })
    }

    /// Apply forward difference (for pressure → velocity)
    pub fn apply_forward_x(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();

        if nx < 2 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 2,
                actual: nx,
                direction: "X".to_string(),
            }
            .into());
        }

        let mut result = Array3::zeros((nx - 1, ny, nz));

        for i in 0..nx - 1 {
            for j in 0..ny {
                for k in 0..nz {
                    result[[i, j, k]] = (field[[i + 1, j, k]] - field[[i, j, k]]) / self.dx;
                }
            }
        }

        Ok(result)
    }

    /// Apply backward difference (for velocity → pressure)
    pub fn apply_backward_x(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();

        if nx < 2 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 2,
                actual: nx,
                direction: "X".to_string(),
            }
            .into());
        }

        let mut result = Array3::zeros((nx, ny, nz));

        for i in 1..nx {
            for j in 0..ny {
                for k in 0..nz {
                    result[[i, j, k]] = (field[[i, j, k]] - field[[i - 1, j, k]]) / self.dx;
                }
            }
        }

        // Boundary at i=0 uses forward difference
        for j in 0..ny {
            for k in 0..nz {
                if nx > 1 {
                    result[[0, j, k]] = (field[[1, j, k]] - field[[0, j, k]]) / self.dx;
                }
            }
        }

        Ok(result)
    }
}

impl DifferentialOperator for StaggeredGridOperator {
    fn apply_x(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        // Default to forward difference for general interface
        self.apply_forward_x(field)
    }

    fn apply_y(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();

        if ny < 2 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 2,
                actual: ny,
                direction: "Y".to_string(),
            }
            .into());
        }

        let mut result = Array3::zeros((nx, ny - 1, nz));

        for i in 0..nx {
            for j in 0..ny - 1 {
                for k in 0..nz {
                    result[[i, j, k]] = (field[[i, j + 1, k]] - field[[i, j, k]]) / self.dy;
                }
            }
        }

        Ok(result)
    }

    fn apply_z(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();

        if nz < 2 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 2,
                actual: nz,
                direction: "Z".to_string(),
            }
            .into());
        }

        let mut result = Array3::zeros((nx, ny, nz - 1));

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz - 1 {
                    result[[i, j, k]] = (field[[i, j, k + 1]] - field[[i, j, k]]) / self.dz;
                }
            }
        }

        Ok(result)
    }

    fn order(&self) -> usize {
        2
    }

    fn stencil_width(&self) -> usize {
        2
    }

    fn is_conservative(&self) -> bool {
        true // Staggered grids preserve discrete conservation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array3;

    #[test]
    fn test_central_difference_2_linear_function() {
        // Test on linear function: u(x,y,z) = 2x + 3y + 4z
        // Exact derivatives: du/dx = 2, du/dy = 3, du/dz = 4
        let dx = 0.1;
        let op = CentralDifference2::new(dx, dx, dx).unwrap();

        let mut field = Array3::zeros((10, 10, 10));
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    field[[i, j, k]] =
                        2.0 * (i as f64) * dx + 3.0 * (j as f64) * dx + 4.0 * (k as f64) * dx;
                }
            }
        }

        let grad_x = op.apply_x(field.view()).unwrap();
        let grad_y = op.apply_y(field.view()).unwrap();
        let grad_z = op.apply_z(field.view()).unwrap();

        // Check interior points (exact for linear functions)
        for i in 1..9 {
            for j in 0..10 {
                for k in 0..10 {
                    assert_abs_diff_eq!(grad_x[[i, j, k]], 2.0, epsilon = 1e-10);
                }
            }
        }

        for i in 0..10 {
            for j in 1..9 {
                for k in 0..10 {
                    assert_abs_diff_eq!(grad_y[[i, j, k]], 3.0, epsilon = 1e-10);
                }
            }
        }

        for i in 0..10 {
            for j in 0..10 {
                for k in 1..9 {
                    assert_abs_diff_eq!(grad_z[[i, j, k]], 4.0, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_central_difference_4_quadratic_function() {
        // Test on quadratic: u(x) = x²
        // du/dx = 2x (exact for 4th order on smooth functions)
        let dx = 0.1;
        let op = CentralDifference4::new(dx, dx, dx).unwrap();

        let mut field = Array3::zeros((20, 5, 5));
        for i in 0..20 {
            let x = (i as f64) * dx;
            for j in 0..5 {
                for k in 0..5 {
                    field[[i, j, k]] = x * x;
                }
            }
        }

        let grad_x = op.apply_x(field.view()).unwrap();

        // Check interior points
        for i in 5..15 {
            let x = (i as f64) * dx;
            let expected = 2.0 * x;
            assert_abs_diff_eq!(grad_x[[i, 2, 2]], expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_staggered_grid_constant_field() {
        let dx = 0.01;
        let op = StaggeredGridOperator::new(dx, dx, dx).unwrap();

        let field = Array3::from_elem((10, 10, 10), 5.0);

        let grad_x = op.apply_forward_x(field.view()).unwrap();

        // Derivative of constant should be zero
        for i in 0..9 {
            for j in 0..10 {
                for k in 0..10 {
                    assert_abs_diff_eq!(grad_x[[i, j, k]], 0.0, epsilon = 1e-15);
                }
            }
        }
    }

    #[test]
    fn test_invalid_grid_spacing() {
        assert!(CentralDifference2::new(0.0, 0.1, 0.1).is_err());
        assert!(CentralDifference2::new(-0.1, 0.1, 0.1).is_err());
    }

    #[test]
    fn test_insufficient_grid_points() {
        let op = CentralDifference2::new(0.1, 0.1, 0.1).unwrap();
        let small_field = Array3::zeros((2, 10, 10));

        // Should fail: need at least 3 points for central difference
        assert!(op.apply_x(small_field.view()).is_err());
    }
}
