//! # Second-Order Central Difference Operator
//!
//! This module implements the second-order accurate central difference scheme
//! for computing spatial derivatives on uniform Cartesian grids.
//!
//! ## Mathematical Specification
//!
//! For a smooth function u(x), the first derivative is approximated by:
//!
//! ```text
//! du/dx ≈ (u[i+1] - u[i-1]) / (2Δx) + O(Δx²)
//! ```
//!
//! ## Stencil
//!
//! Interior points use a 3-point stencil:
//! ```text
//! [-1, 0, +1] with coefficients [-1/2Δx, 0, +1/2Δx]
//! ```
//!
//! Boundary points use first-order forward/backward differences:
//! ```text
//! Left boundary:  (u[1] - u[0]) / Δx
//! Right boundary: (u[n-1] - u[n-2]) / Δx
//! ```
//!
//! ## Properties
//!
//! - **Order**: 2 (interior), 1 (boundaries)
//! - **Stencil Width**: 3 points
//! - **Conservation**: No (standard central difference)
//! - **Adjoint Consistency**: Yes (symmetric stencil)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use kwavers::math::numerics::operators::differential::{DifferentialOperator, CentralDifference2};
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
//!   DOI: 10.1090/S0025-5718-1988-0935077-0

use crate::core::error::{KwaversResult, NumericalError};
use ndarray::{Array3, ArrayView3};

use super::DifferentialOperator;

/// Second-order central difference operator
///
/// This operator computes spatial derivatives using the classical second-order
/// central difference formula. It is the most commonly used finite difference
/// scheme due to its simplicity and adequate accuracy for many applications.
///
/// # Grid Spacing
///
/// The operator requires uniform grid spacing in each direction. Grid spacings
/// can differ between directions (anisotropic grids), but must be constant
/// within each direction.
///
/// # Boundary Treatment
///
/// Boundaries are handled using first-order forward/backward differences,
/// which reduces the global order of accuracy from 2 to 1 near boundaries.
/// For applications requiring higher accuracy at boundaries, consider using
/// higher-order operators or specialized boundary stencils.
#[derive(Debug, Clone)]
pub struct CentralDifference2 {
    /// Grid spacing in X direction (meters)
    dx: f64,
    /// Grid spacing in Y direction (meters)
    dy: f64,
    /// Grid spacing in Z direction (meters)
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
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let op = CentralDifference2::new(0.001, 0.001, 0.001)?; // Isotropic 1mm grid
    /// let op = CentralDifference2::new(0.001, 0.001, 0.002)?; // Anisotropic grid
    /// ```
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

        // Interior points: second-order central difference
        // ∂u/∂x ≈ (u[i+1] - u[i-1]) / (2Δx)
        for i in 1..nx - 1 {
            for j in 0..ny {
                for k in 0..nz {
                    result[[i, j, k]] =
                        (field[[i + 1, j, k]] - field[[i - 1, j, k]]) / (2.0 * self.dx);
                }
            }
        }

        // Boundary points: first-order forward/backward difference
        for j in 0..ny {
            for k in 0..nz {
                // Left boundary: forward difference
                // ∂u/∂x ≈ (u[1] - u[0]) / Δx
                result[[0, j, k]] = (field[[1, j, k]] - field[[0, j, k]]) / self.dx;

                // Right boundary: backward difference
                // ∂u/∂x ≈ (u[n-1] - u[n-2]) / Δx
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

        // Interior points: second-order central difference
        for i in 0..nx {
            for j in 1..ny - 1 {
                for k in 0..nz {
                    result[[i, j, k]] =
                        (field[[i, j + 1, k]] - field[[i, j - 1, k]]) / (2.0 * self.dy);
                }
            }
        }

        // Boundary points: first-order forward/backward difference
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

        // Interior points: second-order central difference
        for i in 0..nx {
            for j in 0..ny {
                for k in 1..nz - 1 {
                    result[[i, j, k]] =
                        (field[[i, j, k + 1]] - field[[i, j, k - 1]]) / (2.0 * self.dz);
                }
            }
        }

        // Boundary points: first-order forward/backward difference
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
        true // Symmetric stencil implies adjoint consistency
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_constructor_valid() {
        let op = CentralDifference2::new(0.1, 0.1, 0.1);
        assert!(op.is_ok());
    }

    #[test]
    fn test_constructor_invalid_spacing() {
        assert!(CentralDifference2::new(0.0, 0.1, 0.1).is_err());
        assert!(CentralDifference2::new(-0.1, 0.1, 0.1).is_err());
        assert!(CentralDifference2::new(0.1, 0.0, 0.1).is_err());
        assert!(CentralDifference2::new(0.1, 0.1, -0.1).is_err());
    }

    #[test]
    fn test_apply_x_linear_function() {
        // Test on linear function: u(x,y,z) = 2x
        // Exact derivative: du/dx = 2
        let dx = 0.1;
        let op = CentralDifference2::new(dx, dx, dx).unwrap();

        let mut field = Array3::zeros((10, 5, 5));
        for i in 0..10 {
            for j in 0..5 {
                for k in 0..5 {
                    field[[i, j, k]] = 2.0 * (i as f64) * dx;
                }
            }
        }

        let grad_x = op.apply_x(field.view()).unwrap();

        // Check interior points (exact for linear functions)
        for i in 1..9 {
            for j in 0..5 {
                for k in 0..5 {
                    assert_abs_diff_eq!(grad_x[[i, j, k]], 2.0, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_apply_y_linear_function() {
        let dy = 0.1;
        let op = CentralDifference2::new(dy, dy, dy).unwrap();

        let mut field = Array3::zeros((5, 10, 5));
        for i in 0..5 {
            for j in 0..10 {
                for k in 0..5 {
                    field[[i, j, k]] = 3.0 * (j as f64) * dy;
                }
            }
        }

        let grad_y = op.apply_y(field.view()).unwrap();

        for i in 0..5 {
            for j in 1..9 {
                for k in 0..5 {
                    assert_abs_diff_eq!(grad_y[[i, j, k]], 3.0, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_apply_z_linear_function() {
        let dz = 0.1;
        let op = CentralDifference2::new(dz, dz, dz).unwrap();

        let mut field = Array3::zeros((5, 5, 10));
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..10 {
                    field[[i, j, k]] = 4.0 * (k as f64) * dz;
                }
            }
        }

        let grad_z = op.apply_z(field.view()).unwrap();

        for i in 0..5 {
            for j in 0..5 {
                for k in 1..9 {
                    assert_abs_diff_eq!(grad_z[[i, j, k]], 4.0, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_constant_field_has_zero_derivative() {
        let op = CentralDifference2::new(0.1, 0.1, 0.1).unwrap();
        let field = Array3::from_elem((10, 10, 10), 5.0);

        let grad_x = op.apply_x(field.view()).unwrap();
        let grad_y = op.apply_y(field.view()).unwrap();
        let grad_z = op.apply_z(field.view()).unwrap();

        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    assert_abs_diff_eq!(grad_x[[i, j, k]], 0.0, epsilon = 1e-15);
                    assert_abs_diff_eq!(grad_y[[i, j, k]], 0.0, epsilon = 1e-15);
                    assert_abs_diff_eq!(grad_z[[i, j, k]], 0.0, epsilon = 1e-15);
                }
            }
        }
    }

    #[test]
    fn test_insufficient_grid_points() {
        let op = CentralDifference2::new(0.1, 0.1, 0.1).unwrap();

        let field_x = Array3::zeros((2, 10, 10));
        let field_y = Array3::zeros((10, 2, 10));
        let field_z = Array3::zeros((10, 10, 2));

        assert!(op.apply_x(field_x.view()).is_err());
        assert!(op.apply_y(field_y.view()).is_err());
        assert!(op.apply_z(field_z.view()).is_err());
    }

    #[test]
    fn test_properties() {
        let op = CentralDifference2::new(0.1, 0.1, 0.1).unwrap();
        assert_eq!(op.order(), 2);
        assert_eq!(op.stencil_width(), 3);
        assert!(op.is_adjoint_consistent());
        assert!(!op.is_conservative());
    }
}
