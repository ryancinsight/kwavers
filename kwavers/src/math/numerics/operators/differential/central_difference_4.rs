//! # Fourth-Order Central Difference Operator
//!
//! This module implements the fourth-order accurate central difference scheme
//! for computing spatial derivatives on uniform Cartesian grids.
//!
//! ## Mathematical Specification
//!
//! For a smooth function u(x), the first derivative is approximated by:
//!
//! ```text
//! du/dx ≈ (-u[i+2] + 8u[i+1] - 8u[i-1] + u[i-2]) / (12Δx) + O(Δx⁴)
//! ```
//!
//! ## Stencil
//!
//! Interior points use a 5-point stencil:
//! ```text
//! [-2, -1, 0, +1, +2] with coefficients [1/12Δx, -8/12Δx, 0, 8/12Δx, -1/12Δx]
//! ```
//!
//! Near-boundary points use second-order central difference:
//! ```text
//! (u[i+1] - u[i-1]) / (2Δx)
//! ```
//!
//! Boundary points use first-order forward/backward differences.
//!
//! ## Properties
//!
//! - **Order**: 4 (interior), 2 (near-boundary), 1 (boundaries)
//! - **Stencil Width**: 5 points
//! - **Conservation**: No (standard central difference)
//! - **Adjoint Consistency**: Yes (symmetric stencil)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use kwavers::math::numerics::operators::differential::{DifferentialOperator, CentralDifference4};
//! use ndarray::Array3;
//!
//! let dx = 0.001; // 1 mm grid spacing
//! let op = CentralDifference4::new(dx, dx, dx)?;
//!
//! let field = Array3::zeros((100, 100, 100));
//! let gradient_x = op.apply_x(field.view())?;
//! ```
//!
//! ## References
//!
//! - Shubin, G. R., & Bell, J. B. (1987). "A modified equation approach to
//!   constructing fourth order methods for acoustic wave propagation."
//!   *SIAM Journal on Scientific and Statistical Computing*, 8(2), 135-151.
//!   DOI: 10.1137/0908025

use crate::core::error::{KwaversResult, NumericalError};
use ndarray::{Array3, ArrayView3};

use super::DifferentialOperator;

/// Fourth-order central difference operator
///
/// This operator provides higher accuracy than second-order schemes while
/// maintaining computational efficiency. It is commonly used in wave propagation
/// simulations where dispersion error must be minimized.
///
/// # Accuracy Considerations
///
/// The fourth-order scheme provides significantly better phase accuracy for
/// wave propagation compared to second-order methods. For acoustic simulations,
/// this translates to:
///
/// - Reduced numerical dispersion
/// - Fewer points per wavelength required
/// - Better long-time accuracy
///
/// # Boundary Treatment
///
/// A multi-order approach is used:
/// - Interior (i ∈ [2, n-3]): Fourth-order central (5-point stencil)
/// - Near-boundary (i = 1, n-2): Second-order central (3-point stencil)
/// - Boundary (i = 0, n-1): First-order forward/backward (2-point stencil)
#[derive(Debug, Clone)]
pub struct CentralDifference4 {
    /// Grid spacing in X direction (meters)
    dx: f64,
    /// Grid spacing in Y direction (meters)
    dy: f64,
    /// Grid spacing in Z direction (meters)
    dz: f64,
}

impl CentralDifference4 {
    /// Create a new fourth-order central difference operator
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
    /// let op = CentralDifference4::new(0.001, 0.001, 0.001)?; // Isotropic 1mm grid
    /// ```
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

        // Interior points: fourth-order central difference
        // ∂u/∂x ≈ (-u[i+2] + 8u[i+1] - 8u[i-1] + u[i-2]) / (12Δx)
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

        // Near-boundary points: second-order central difference
        // ∂u/∂x ≈ (u[i+1] - u[i-1]) / (2Δx)
        for j in 0..ny {
            for k in 0..nz {
                if nx > 2 {
                    result[[1, j, k]] = (field[[2, j, k]] - field[[0, j, k]]) / (2.0 * self.dx);
                    result[[nx - 2, j, k]] =
                        (field[[nx - 1, j, k]] - field[[nx - 3, j, k]]) / (2.0 * self.dx);
                }
            }
        }

        // Boundary points: first-order forward/backward difference
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

        // Interior points: fourth-order central difference
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

        // Near-boundary and boundary points
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

        // Interior points: fourth-order central difference
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
        true // Symmetric stencil implies adjoint consistency
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_constructor_valid() {
        let op = CentralDifference4::new(0.1, 0.1, 0.1);
        assert!(op.is_ok());
    }

    #[test]
    fn test_constructor_invalid_spacing() {
        assert!(CentralDifference4::new(0.0, 0.1, 0.1).is_err());
        assert!(CentralDifference4::new(-0.1, 0.1, 0.1).is_err());
    }

    #[test]
    fn test_apply_x_linear_function() {
        // Fourth-order scheme is exact for linear functions
        let dx = 0.1;
        let op = CentralDifference4::new(dx, dx, dx).unwrap();

        let mut field = Array3::zeros((20, 5, 5));
        for i in 0..20 {
            for j in 0..5 {
                for k in 0..5 {
                    field[[i, j, k]] = 2.0 * (i as f64) * dx;
                }
            }
        }

        let grad_x = op.apply_x(field.view()).unwrap();

        // Check interior points (fourth-order stencil)
        for i in 2..18 {
            for j in 0..5 {
                for k in 0..5 {
                    assert_abs_diff_eq!(grad_x[[i, j, k]], 2.0, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_apply_x_quadratic_function() {
        // Test on quadratic: u(x) = x²
        // du/dx = 2x (fourth-order should be very accurate)
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

        // Check interior points with fourth-order accuracy
        for i in 5..15 {
            let x = (i as f64) * dx;
            let expected = 2.0 * x;
            assert_abs_diff_eq!(grad_x[[i, 2, 2]], expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_constant_field_has_zero_derivative() {
        let op = CentralDifference4::new(0.1, 0.1, 0.1).unwrap();
        let field = Array3::from_elem((20, 20, 20), 5.0);

        let grad_x = op.apply_x(field.view()).unwrap();
        let grad_y = op.apply_y(field.view()).unwrap();
        let grad_z = op.apply_z(field.view()).unwrap();

        for i in 0..20 {
            for j in 0..20 {
                for k in 0..20 {
                    assert_abs_diff_eq!(grad_x[[i, j, k]], 0.0, epsilon = 1e-15);
                    assert_abs_diff_eq!(grad_y[[i, j, k]], 0.0, epsilon = 1e-15);
                    assert_abs_diff_eq!(grad_z[[i, j, k]], 0.0, epsilon = 1e-15);
                }
            }
        }
    }

    #[test]
    fn test_insufficient_grid_points() {
        let op = CentralDifference4::new(0.1, 0.1, 0.1).unwrap();

        let field_x = Array3::zeros((4, 10, 10));
        let field_y = Array3::zeros((10, 4, 10));
        let field_z = Array3::zeros((10, 10, 4));

        assert!(op.apply_x(field_x.view()).is_err());
        assert!(op.apply_y(field_y.view()).is_err());
        assert!(op.apply_z(field_z.view()).is_err());
    }

    #[test]
    fn test_properties() {
        let op = CentralDifference4::new(0.1, 0.1, 0.1).unwrap();
        assert_eq!(op.order(), 4);
        assert_eq!(op.stencil_width(), 5);
        assert!(op.is_adjoint_consistent());
        assert!(!op.is_conservative());
    }

    #[test]
    fn test_symmetry() {
        // Test that symmetric input produces symmetric output
        let dx = 0.1;
        let op = CentralDifference4::new(dx, dx, dx).unwrap();

        let mut field = Array3::zeros((20, 5, 5));
        for i in 0..20 {
            let x = (i as f64) * dx - 1.0; // Center at x=1
            for j in 0..5 {
                for k in 0..5 {
                    field[[i, j, k]] = x * x; // Symmetric function
                }
            }
        }

        let grad_x = op.apply_x(field.view()).unwrap();

        // Derivative should be antisymmetric around center
        let center = 10;
        for offset in 1..8 {
            let left = grad_x[[center - offset, 2, 2]];
            let right = grad_x[[center + offset, 2, 2]];
            assert_abs_diff_eq!(left, -right, epsilon = 1e-10);
        }
    }
}
