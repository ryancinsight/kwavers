//! # Sixth-Order Central Difference Operator
//!
//! This module implements the sixth-order accurate central difference scheme
//! for computing spatial derivatives on uniform Cartesian grids.
//!
//! ## Mathematical Specification
//!
//! For a smooth function u(x), the first derivative is approximated by:
//!
//! ```text
//! du/dx ≈ (-u[i+3] + 9u[i+2] - 45u[i+1] + 45u[i-1] - 9u[i-2] + u[i-3]) / (60Δx) + O(Δx⁶)
//! ```
//!
//! ## Stencil
//!
//! Interior points use a 7-point stencil:
//! ```text
//! [-3, -2, -1, 0, +1, +2, +3]
//! with coefficients [1/60Δx, -9/60Δx, 45/60Δx, 0, -45/60Δx, 9/60Δx, -1/60Δx]
//! ```
//!
//! Near-boundary points use progressively lower-order stencils:
//! - i = 2, n-3: Fourth-order (5-point stencil)
//! - i = 1, n-2: Second-order (3-point stencil)
//! - i = 0, n-1: First-order (2-point stencil)
//!
//! ## Properties
//!
//! - **Order**: 6 (interior), 4/2/1 (near-boundary to boundary)
//! - **Stencil Width**: 7 points
//! - **Conservation**: No (standard central difference)
//! - **Adjoint Consistency**: Yes (symmetric stencil)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use kwavers::math::numerics::operators::differential::{DifferentialOperator, CentralDifference6};
//! use ndarray::Array3;
//!
//! let dx = 0.001; // 1 mm grid spacing
//! let op = CentralDifference6::new(dx, dx, dx)?;
//!
//! let field = Array3::zeros((100, 100, 100));
//! let gradient_x = op.apply_x(field.view())?;
//! ```
//!
//! ## When to Use
//!
//! Sixth-order schemes are recommended for:
//! - Long-time wave propagation simulations
//! - High-frequency content requiring minimal dispersion
//! - Applications where accuracy outweighs computational cost
//!
//! ## References
//!
//! - Fornberg, B. (1988). "Generation of finite difference formulas on arbitrarily
//!   spaced grids." *Mathematics of Computation*, 51(184), 699-706.
//!   DOI: 10.1090/S0025-5718-1988-0935077-0

use crate::core::error::{KwaversResult, NumericalError};
use ndarray::{Array3, ArrayView3};

use super::DifferentialOperator;

/// Sixth-order central difference operator
///
/// This operator provides the highest accuracy of the standard central difference
/// schemes available in kwavers. It is particularly effective for wave propagation
/// problems requiring minimal numerical dispersion over long propagation distances.
///
/// # Accuracy Considerations
///
/// The sixth-order scheme provides excellent phase accuracy:
/// - Minimal numerical dispersion for smooth solutions
/// - Approximately 2-3 points per wavelength for accurate propagation
/// - Superior to lower-order methods for long-time integration
///
/// However, it comes with trade-offs:
/// - Requires more memory bandwidth (7-point stencil vs 3 or 5)
/// - More sensitive to discontinuities and sharp gradients
/// - Larger boundary layer with reduced accuracy
///
/// # Boundary Treatment
///
/// A progressive multi-order approach:
/// - Interior (i ∈ [3, n-4]): Sixth-order central (7-point)
/// - Near-boundary 1 (i = 2, n-3): Fourth-order central (5-point)
/// - Near-boundary 2 (i = 1, n-2): Second-order central (3-point)
/// - Boundary (i = 0, n-1): First-order forward/backward (2-point)
#[derive(Debug, Clone)]
pub struct CentralDifference6 {
    /// Grid spacing in X direction (meters)
    dx: f64,
    /// Grid spacing in Y direction (meters)
    dy: f64,
    /// Grid spacing in Z direction (meters)
    dz: f64,
}

impl CentralDifference6 {
    /// Create a new sixth-order central difference operator
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
    /// let op = CentralDifference6::new(0.001, 0.001, 0.001)?; // Isotropic 1mm grid
    /// ```
    pub fn new(dx: f64, dy: f64, dz: f64) -> KwaversResult<Self> {
        if dx <= 0.0 || dy <= 0.0 || dz <= 0.0 {
            return Err(NumericalError::InvalidGridSpacing { dx, dy, dz }.into());
        }
        Ok(Self { dx, dy, dz })
    }

    /// Apply ∂/∂x into a pre-allocated destination — zero heap allocation.
    ///
    /// Interior stencil (O(Δx⁶)):
    /// ```text
    /// dst[i] = (−f[i−3] + 9f[i−2] − 45f[i−1] + 45f[i+1] − 9f[i+2] + f[i+3]) / (60 Δx)
    /// ```
    /// Near-boundary: O(Δx⁴) at i=2/n−3, O(Δx²) at i=1/n−2, O(Δx) at i=0/n−1.
    pub fn apply_x_into(&self, field: ArrayView3<f64>, dst: &mut Array3<f64>) -> KwaversResult<()> {
        let (nx, ny, nz) = field.dim();
        if nx < 7 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 7,
                actual: nx,
                direction: "X".to_string(),
            }
            .into());
        }
        let inv60dx = 1.0 / (60.0 * self.dx);
        let inv12dx = 1.0 / (12.0 * self.dx);
        let inv2dx = 1.0 / (2.0 * self.dx);
        let invdx = 1.0 / self.dx;
        for i in 3..nx - 3 {
            for j in 0..ny {
                for k in 0..nz {
                    dst[[i, j, k]] = (-field[[i - 3, j, k]] + 9.0 * field[[i - 2, j, k]]
                        - 45.0 * field[[i - 1, j, k]]
                        + 45.0 * field[[i + 1, j, k]]
                        - 9.0 * field[[i + 2, j, k]]
                        + field[[i + 3, j, k]])
                        * inv60dx;
                }
            }
        }
        for j in 0..ny {
            for k in 0..nz {
                dst[[2, j, k]] = (-field[[4, j, k]] + 8.0 * field[[3, j, k]]
                    - 8.0 * field[[1, j, k]]
                    + field[[0, j, k]])
                    * inv12dx;
                dst[[nx - 3, j, k]] = (-field[[nx - 1, j, k]] + 8.0 * field[[nx - 2, j, k]]
                    - 8.0 * field[[nx - 4, j, k]]
                    + field[[nx - 5, j, k]])
                    * inv12dx;
                dst[[1, j, k]] = (field[[2, j, k]] - field[[0, j, k]]) * inv2dx;
                dst[[nx - 2, j, k]] = (field[[nx - 1, j, k]] - field[[nx - 3, j, k]]) * inv2dx;
                dst[[0, j, k]] = (field[[1, j, k]] - field[[0, j, k]]) * invdx;
                dst[[nx - 1, j, k]] = (field[[nx - 1, j, k]] - field[[nx - 2, j, k]]) * invdx;
            }
        }
        Ok(())
    }

    /// Apply ∂/∂y into a pre-allocated destination — zero heap allocation.
    pub fn apply_y_into(&self, field: ArrayView3<f64>, dst: &mut Array3<f64>) -> KwaversResult<()> {
        let (nx, ny, nz) = field.dim();
        if ny < 7 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 7,
                actual: ny,
                direction: "Y".to_string(),
            }
            .into());
        }
        let inv60dy = 1.0 / (60.0 * self.dy);
        let inv12dy = 1.0 / (12.0 * self.dy);
        let inv2dy = 1.0 / (2.0 * self.dy);
        let invdy = 1.0 / self.dy;
        for i in 0..nx {
            for j in 3..ny - 3 {
                for k in 0..nz {
                    dst[[i, j, k]] = (-field[[i, j - 3, k]] + 9.0 * field[[i, j - 2, k]]
                        - 45.0 * field[[i, j - 1, k]]
                        + 45.0 * field[[i, j + 1, k]]
                        - 9.0 * field[[i, j + 2, k]]
                        + field[[i, j + 3, k]])
                        * inv60dy;
                }
            }
        }
        for i in 0..nx {
            for k in 0..nz {
                dst[[i, 2, k]] = (-field[[i, 4, k]] + 8.0 * field[[i, 3, k]]
                    - 8.0 * field[[i, 1, k]]
                    + field[[i, 0, k]])
                    * inv12dy;
                dst[[i, ny - 3, k]] = (-field[[i, ny - 1, k]] + 8.0 * field[[i, ny - 2, k]]
                    - 8.0 * field[[i, ny - 4, k]]
                    + field[[i, ny - 5, k]])
                    * inv12dy;
                dst[[i, 1, k]] = (field[[i, 2, k]] - field[[i, 0, k]]) * inv2dy;
                dst[[i, ny - 2, k]] = (field[[i, ny - 1, k]] - field[[i, ny - 3, k]]) * inv2dy;
                dst[[i, 0, k]] = (field[[i, 1, k]] - field[[i, 0, k]]) * invdy;
                dst[[i, ny - 1, k]] = (field[[i, ny - 1, k]] - field[[i, ny - 2, k]]) * invdy;
            }
        }
        Ok(())
    }

    /// Apply ∂/∂z into a pre-allocated destination — zero heap allocation.
    pub fn apply_z_into(&self, field: ArrayView3<f64>, dst: &mut Array3<f64>) -> KwaversResult<()> {
        let (nx, ny, nz) = field.dim();
        if nz < 7 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 7,
                actual: nz,
                direction: "Z".to_string(),
            }
            .into());
        }
        let inv60dz = 1.0 / (60.0 * self.dz);
        let inv12dz = 1.0 / (12.0 * self.dz);
        let inv2dz = 1.0 / (2.0 * self.dz);
        let invdz = 1.0 / self.dz;
        for i in 0..nx {
            for j in 0..ny {
                for k in 3..nz - 3 {
                    dst[[i, j, k]] = (-field[[i, j, k - 3]] + 9.0 * field[[i, j, k - 2]]
                        - 45.0 * field[[i, j, k - 1]]
                        + 45.0 * field[[i, j, k + 1]]
                        - 9.0 * field[[i, j, k + 2]]
                        + field[[i, j, k + 3]])
                        * inv60dz;
                }
            }
        }
        for i in 0..nx {
            for j in 0..ny {
                dst[[i, j, 2]] = (-field[[i, j, 4]] + 8.0 * field[[i, j, 3]]
                    - 8.0 * field[[i, j, 1]]
                    + field[[i, j, 0]])
                    * inv12dz;
                dst[[i, j, nz - 3]] = (-field[[i, j, nz - 1]] + 8.0 * field[[i, j, nz - 2]]
                    - 8.0 * field[[i, j, nz - 4]]
                    + field[[i, j, nz - 5]])
                    * inv12dz;
                dst[[i, j, 1]] = (field[[i, j, 2]] - field[[i, j, 0]]) * inv2dz;
                dst[[i, j, nz - 2]] = (field[[i, j, nz - 1]] - field[[i, j, nz - 3]]) * inv2dz;
                dst[[i, j, 0]] = (field[[i, j, 1]] - field[[i, j, 0]]) * invdz;
                dst[[i, j, nz - 1]] = (field[[i, j, nz - 1]] - field[[i, j, nz - 2]]) * invdz;
            }
        }
        Ok(())
    }
}

impl DifferentialOperator for CentralDifference6 {
    fn apply_x(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();
        if nx < 7 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 7,
                actual: nx,
                direction: "X".to_string(),
            }
            .into());
        }
        let mut result = Array3::zeros((nx, ny, nz));
        self.apply_x_into(field, &mut result)?;
        Ok(result)
    }

    fn apply_y(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();
        if ny < 7 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 7,
                actual: ny,
                direction: "Y".to_string(),
            }
            .into());
        }
        let mut result = Array3::zeros((nx, ny, nz));
        self.apply_y_into(field, &mut result)?;
        Ok(result)
    }

    fn apply_z(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();
        if nz < 7 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 7,
                actual: nz,
                direction: "Z".to_string(),
            }
            .into());
        }
        let mut result = Array3::zeros((nx, ny, nz));
        self.apply_z_into(field, &mut result)?;
        Ok(result)
    }

    fn order(&self) -> usize {
        6
    }

    fn stencil_width(&self) -> usize {
        7
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
        let op = CentralDifference6::new(0.1, 0.1, 0.1);
        assert!(op.is_ok());
    }

    #[test]
    fn test_constructor_invalid_spacing() {
        assert!(CentralDifference6::new(0.0, 0.1, 0.1).is_err());
        assert!(CentralDifference6::new(-0.1, 0.1, 0.1).is_err());
    }

    #[test]
    fn test_apply_x_linear_function() {
        // Sixth-order scheme is exact for linear functions
        let dx = 0.1;
        let op = CentralDifference6::new(dx, dx, dx).unwrap();

        let mut field = Array3::zeros((12, 5, 5));
        for i in 0..12 {
            for j in 0..5 {
                for k in 0..5 {
                    field[[i, j, k]] = 2.0 * (i as f64) * dx;
                }
            }
        }

        let grad_x = op.apply_x(field.view()).unwrap();

        // Check interior points (sixth-order stencil)
        for i in 3..9 {
            for j in 0..5 {
                for k in 0..5 {
                    assert_abs_diff_eq!(grad_x[[i, j, k]], 2.0, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_all_directions_linear_function() {
        let dx = 0.1;
        let op = CentralDifference6::new(dx, dx, dx).unwrap();

        let mut field = Array3::zeros((12, 12, 12));
        for i in 0..12 {
            for j in 0..12 {
                for k in 0..12 {
                    field[[i, j, k]] =
                        2.0 * (i as f64) * dx + 3.0 * (j as f64) * dx + 4.0 * (k as f64) * dx;
                }
            }
        }

        let grad_x = op.apply_x(field.view()).unwrap();
        let grad_y = op.apply_y(field.view()).unwrap();
        let grad_z = op.apply_z(field.view()).unwrap();

        // Check interior points in all directions
        for i in 3..9 {
            for j in 3..9 {
                for k in 3..9 {
                    assert_abs_diff_eq!(grad_x[[i, j, k]], 2.0, epsilon = 1e-10);
                    assert_abs_diff_eq!(grad_y[[i, j, k]], 3.0, epsilon = 1e-10);
                    assert_abs_diff_eq!(grad_z[[i, j, k]], 4.0, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_constant_field_has_zero_derivative() {
        let op = CentralDifference6::new(0.1, 0.1, 0.1).unwrap();
        let field = Array3::from_elem((12, 12, 12), 5.0);

        let grad_x = op.apply_x(field.view()).unwrap();
        let grad_y = op.apply_y(field.view()).unwrap();
        let grad_z = op.apply_z(field.view()).unwrap();

        for i in 0..12 {
            for j in 0..12 {
                for k in 0..12 {
                    assert_abs_diff_eq!(grad_x[[i, j, k]], 0.0, epsilon = 1e-15);
                    assert_abs_diff_eq!(grad_y[[i, j, k]], 0.0, epsilon = 1e-15);
                    assert_abs_diff_eq!(grad_z[[i, j, k]], 0.0, epsilon = 1e-15);
                }
            }
        }
    }

    #[test]
    fn test_insufficient_grid_points() {
        let op = CentralDifference6::new(0.1, 0.1, 0.1).unwrap();

        let field_x = Array3::zeros((6, 10, 10));
        let field_y = Array3::zeros((10, 6, 10));
        let field_z = Array3::zeros((10, 10, 6));

        assert!(op.apply_x(field_x.view()).is_err());
        assert!(op.apply_y(field_y.view()).is_err());
        assert!(op.apply_z(field_z.view()).is_err());
    }

    #[test]
    fn test_properties() {
        let op = CentralDifference6::new(0.1, 0.1, 0.1).unwrap();
        assert_eq!(op.order(), 6);
        assert_eq!(op.stencil_width(), 7);
        assert!(op.is_adjoint_consistent());
        assert!(!op.is_conservative());
    }

    #[test]
    fn test_cubic_polynomial() {
        // Test on cubic: u(x) = x³
        // du/dx = 3x²
        let dx = 0.1;
        let op = CentralDifference6::new(dx, dx, dx).unwrap();

        let mut field = Array3::zeros((20, 5, 5));
        for i in 0..20 {
            let x = (i as f64) * dx;
            for j in 0..5 {
                for k in 0..5 {
                    field[[i, j, k]] = x * x * x;
                }
            }
        }

        let grad_x = op.apply_x(field.view()).unwrap();

        // Check interior points with sixth-order accuracy
        for i in 5..15 {
            let x = (i as f64) * dx;
            let expected = 3.0 * x * x;
            assert_abs_diff_eq!(grad_x[[i, 2, 2]], expected, epsilon = 1e-8);
        }
    }
}
