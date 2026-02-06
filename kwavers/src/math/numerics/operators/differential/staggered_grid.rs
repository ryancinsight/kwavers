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
//! ## Usage
//!
//! ```rust,ignore
//! use kwavers::math::numerics::operators::differential::{StaggeredGridOperator};
//! use ndarray::Array3;
//!
//! let dx = 0.001; // 1 mm grid spacing
//! let op = StaggeredGridOperator::new(dx, dx, dx)?;
//!
//! // Forward difference (for FDTD pressure → velocity update)
//! let grad_forward = op.apply_forward_x(pressure.view())?;
//!
//! // Backward difference (for FDTD velocity → pressure update)
//! let grad_backward = op.apply_backward_x(velocity.view())?;
//! ```
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

use crate::core::error::{KwaversResult, NumericalError};
use ndarray::{Array3, ArrayView3};

use super::DifferentialOperator;

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
    dx: f64,
    /// Grid spacing in Y direction (meters)
    dy: f64,
    /// Grid spacing in Z direction (meters)
    dz: f64,
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
    /// # Returns
    ///
    /// New operator instance
    ///
    /// # Errors
    ///
    /// Returns error if any grid spacing is non-positive
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let op = StaggeredGridOperator::new(0.001, 0.001, 0.001)?; // Isotropic 1mm grid
    /// ```
    pub fn new(dx: f64, dy: f64, dz: f64) -> KwaversResult<Self> {
        if dx <= 0.0 || dy <= 0.0 || dz <= 0.0 {
            return Err(NumericalError::InvalidGridSpacing { dx, dy, dz }.into());
        }
        Ok(Self { dx, dy, dz })
    }

    /// Apply forward difference in X direction
    ///
    /// Computes derivative at cell edges (i+1/2) from cell centers (i, i+1).
    /// Used for pressure → velocity updates in FDTD.
    ///
    /// # Arguments
    ///
    /// * `field` - Input field at cell centers (nx × ny × nz)
    ///
    /// # Returns
    ///
    /// Derivative at cell edges ((nx-1) × ny × nz)
    ///
    /// # Mathematical Specification
    ///
    /// ```text
    /// ∂u/∂x|_{i+1/2,j,k} ≈ (u[i+1,j,k] - u[i,j,k]) / Δx
    /// ```
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

    /// Apply backward difference in X direction
    ///
    /// Computes derivative at cell centers (i) from cell edges (i-1, i).
    /// Used for velocity → pressure updates in FDTD.
    ///
    /// # Arguments
    ///
    /// * `field` - Input field at cell edges (nx × ny × nz)
    ///
    /// # Returns
    ///
    /// Derivative at cell centers (nx × ny × nz)
    ///
    /// # Mathematical Specification
    ///
    /// ```text
    /// ∂u/∂x|_{i,j,k} ≈ (u[i,j,k] - u[i-1,j,k]) / Δx
    /// ```
    ///
    /// # Boundary Treatment
    ///
    /// At i=0, uses forward difference since no i-1 point exists.
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

        // Interior points: backward difference
        for i in 1..nx {
            for j in 0..ny {
                for k in 0..nz {
                    result[[i, j, k]] = (field[[i, j, k]] - field[[i - 1, j, k]]) / self.dx;
                }
            }
        }

        // Boundary at i=0: use forward difference
        for j in 0..ny {
            for k in 0..nz {
                if nx > 1 {
                    result[[0, j, k]] = (field[[1, j, k]] - field[[0, j, k]]) / self.dx;
                }
            }
        }

        Ok(result)
    }

    /// Apply backward difference in Y direction
    ///
    /// Computes derivative at cell centers from cell edges in Y direction.
    ///
    /// # Arguments
    ///
    /// * `field` - Input field (nx × ny × nz)
    ///
    /// # Returns
    ///
    /// Derivative at cell centers (nx × ny × nz)
    pub fn apply_backward_y(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();

        if ny < 2 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 2,
                actual: ny,
                direction: "Y".to_string(),
            }
            .into());
        }

        let mut result = Array3::zeros((nx, ny, nz));

        // Interior points: backward difference
        for i in 0..nx {
            for j in 1..ny {
                for k in 0..nz {
                    result[[i, j, k]] = (field[[i, j, k]] - field[[i, j - 1, k]]) / self.dy;
                }
            }
        }

        // Boundary at j=0: use forward difference
        for i in 0..nx {
            for k in 0..nz {
                if ny > 1 {
                    result[[i, 0, k]] = (field[[i, 1, k]] - field[[i, 0, k]]) / self.dy;
                }
            }
        }

        Ok(result)
    }

    /// Apply backward difference in Z direction
    ///
    /// Computes derivative at cell centers from cell edges in Z direction.
    ///
    /// # Arguments
    ///
    /// * `field` - Input field (nx × ny × nz)
    ///
    /// # Returns
    ///
    /// Derivative at cell centers (nx × ny × nz)
    pub fn apply_backward_z(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();

        if nz < 2 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 2,
                actual: nz,
                direction: "Z".to_string(),
            }
            .into());
        }

        let mut result = Array3::zeros((nx, ny, nz));

        // Interior points: backward difference
        for i in 0..nx {
            for j in 0..ny {
                for k in 1..nz {
                    result[[i, j, k]] = (field[[i, j, k]] - field[[i, j, k - 1]]) / self.dz;
                }
            }
        }

        // Boundary at k=0: use forward difference
        for i in 0..nx {
            for j in 0..ny {
                if nz > 1 {
                    result[[i, j, 0]] = (field[[i, j, 1]] - field[[i, j, 0]]) / self.dz;
                }
            }
        }

        Ok(result)
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

    /// Apply operator in Z direction (default: forward difference)
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
        true // Staggered grids preserve discrete conservation laws
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_constructor_valid() {
        let op = StaggeredGridOperator::new(0.1, 0.1, 0.1);
        assert!(op.is_ok());
    }

    #[test]
    fn test_constructor_invalid_spacing() {
        assert!(StaggeredGridOperator::new(0.0, 0.1, 0.1).is_err());
        assert!(StaggeredGridOperator::new(-0.1, 0.1, 0.1).is_err());
    }

    #[test]
    fn test_forward_difference_linear_function() {
        let dx = 0.1;
        let op = StaggeredGridOperator::new(dx, dx, dx).unwrap();

        let mut field = Array3::zeros((10, 5, 5));
        for i in 0..10 {
            for j in 0..5 {
                for k in 0..5 {
                    field[[i, j, k]] = 2.0 * (i as f64) * dx;
                }
            }
        }

        let grad = op.apply_forward_x(field.view()).unwrap();

        // Forward difference of linear function is exact
        assert_eq!(grad.dim(), (9, 5, 5)); // nx-1 points
        for i in 0..9 {
            for j in 0..5 {
                for k in 0..5 {
                    assert_abs_diff_eq!(grad[[i, j, k]], 2.0, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_backward_difference_linear_function() {
        let dx = 0.1;
        let op = StaggeredGridOperator::new(dx, dx, dx).unwrap();

        let mut field = Array3::zeros((10, 5, 5));
        for i in 0..10 {
            for j in 0..5 {
                for k in 0..5 {
                    field[[i, j, k]] = 2.0 * (i as f64) * dx;
                }
            }
        }

        let grad = op.apply_backward_x(field.view()).unwrap();

        // Backward difference of linear function is exact (except boundary)
        assert_eq!(grad.dim(), (10, 5, 5)); // Same size
        for i in 1..10 {
            for j in 0..5 {
                for k in 0..5 {
                    assert_abs_diff_eq!(grad[[i, j, k]], 2.0, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_constant_field_has_zero_derivative() {
        let op = StaggeredGridOperator::new(0.01, 0.01, 0.01).unwrap();
        let field = Array3::from_elem((10, 10, 10), 5.0);

        let grad_forward = op.apply_forward_x(field.view()).unwrap();
        let grad_backward = op.apply_backward_x(field.view()).unwrap();

        // Derivative of constant should be zero
        for i in 0..9 {
            for j in 0..10 {
                for k in 0..10 {
                    assert_abs_diff_eq!(grad_forward[[i, j, k]], 0.0, epsilon = 1e-15);
                }
            }
        }

        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    assert_abs_diff_eq!(grad_backward[[i, j, k]], 0.0, epsilon = 1e-15);
                }
            }
        }
    }

    #[test]
    fn test_insufficient_grid_points() {
        let op = StaggeredGridOperator::new(0.1, 0.1, 0.1).unwrap();

        let field = Array3::zeros((1, 10, 10));

        assert!(op.apply_forward_x(field.view()).is_err());
        assert!(op.apply_backward_x(field.view()).is_err());
    }

    #[test]
    fn test_properties() {
        let op = StaggeredGridOperator::new(0.1, 0.1, 0.1).unwrap();
        assert_eq!(op.order(), 2);
        assert_eq!(op.stencil_width(), 2);
        assert!(op.is_conservative());
        assert!(op.is_adjoint_consistent());
    }

    #[test]
    fn test_forward_backward_complementarity() {
        // Test that forward and backward differences are complementary
        let dx = 0.1;
        let op = StaggeredGridOperator::new(dx, dx, dx).unwrap();

        let mut field = Array3::zeros((10, 5, 5));
        for i in 0..10 {
            let x = (i as f64) * dx;
            for j in 0..5 {
                for k in 0..5 {
                    field[[i, j, k]] = x * x; // Quadratic function
                }
            }
        }

        let grad_forward = op.apply_forward_x(field.view()).unwrap();
        let grad_backward = op.apply_backward_x(field.view()).unwrap();

        // Forward difference at i gives derivative at i+1/2
        // Backward difference at i+1 gives derivative at i+1/2
        // They should be approximately equal
        for i in 0..9 {
            for j in 0..5 {
                for k in 0..5 {
                    let forward_at_half = grad_forward[[i, j, k]];
                    let backward_at_half = grad_backward[[i + 1, j, k]];
                    assert_abs_diff_eq!(forward_at_half, backward_at_half, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_all_directions() {
        let dx = 0.1;
        let op = StaggeredGridOperator::new(dx, dx, dx).unwrap();

        let mut field = Array3::zeros((10, 10, 10));
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    field[[i, j, k]] =
                        2.0 * (i as f64) * dx + 3.0 * (j as f64) * dx + 4.0 * (k as f64) * dx;
                }
            }
        }

        let grad_y = op.apply_backward_y(field.view()).unwrap();
        let grad_z = op.apply_backward_z(field.view()).unwrap();

        // Check derivatives (interior points)
        for i in 0..10 {
            for j in 1..10 {
                for k in 1..10 {
                    assert_abs_diff_eq!(grad_y[[i, j, k]], 3.0, epsilon = 1e-10);
                    assert_abs_diff_eq!(grad_z[[i, j, k]], 4.0, epsilon = 1e-10);
                }
            }
        }
    }
}
