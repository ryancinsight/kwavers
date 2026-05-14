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

#[cfg(test)]
mod tests;

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

impl CentralDifference4 {
    /// Apply ∂/∂x into a pre-allocated destination — zero heap allocation.
    ///
    /// Writes the O(Δx⁴) interior stencil and lower-order boundary stencils
    /// directly into `dst` without any intermediate allocation.
    ///
    /// # Stencil
    ///
    /// Interior (i ∈ [2, nx−3]):
    /// ```text
    /// dst[i] = (−f[i+2] + 8f[i+1] − 8f[i−1] + f[i−2]) / (12 Δx)
    /// ```
    /// Near-boundary (i = 1, nx−2): O(Δx²) central.
    /// Boundary (i = 0, nx−1): O(Δx) one-sided.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn apply_x_into(&self, field: ArrayView3<f64>, dst: &mut Array3<f64>) -> KwaversResult<()> {
        let (nx, ny, nz) = field.dim();
        if nx < 5 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 5,
                actual: nx,
                direction: "X".to_owned(),
            }
            .into());
        }
        let inv12dx = 1.0 / (12.0 * self.dx);
        let inv2dx = 1.0 / (2.0 * self.dx);
        let invdx = 1.0 / self.dx;
        for i in 2..nx - 2 {
            for j in 0..ny {
                for k in 0..nz {
                    dst[[i, j, k]] = (8.0f64.mul_add(
                        -field[[i - 1, j, k]],
                        8.0f64.mul_add(field[[i + 1, j, k]], -field[[i + 2, j, k]]),
                    ) + field[[i - 2, j, k]])
                        * inv12dx;
                }
            }
        }
        for j in 0..ny {
            for k in 0..nz {
                if nx > 2 {
                    dst[[1, j, k]] = (field[[2, j, k]] - field[[0, j, k]]) * inv2dx;
                    dst[[nx - 2, j, k]] = (field[[nx - 1, j, k]] - field[[nx - 3, j, k]]) * inv2dx;
                }
                dst[[0, j, k]] = (field[[1, j, k]] - field[[0, j, k]]) * invdx;
                dst[[nx - 1, j, k]] = (field[[nx - 1, j, k]] - field[[nx - 2, j, k]]) * invdx;
            }
        }
        Ok(())
    }

    /// Apply ∂/∂y into a pre-allocated destination — zero heap allocation.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn apply_y_into(&self, field: ArrayView3<f64>, dst: &mut Array3<f64>) -> KwaversResult<()> {
        let (nx, ny, nz) = field.dim();
        if ny < 5 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 5,
                actual: ny,
                direction: "Y".to_owned(),
            }
            .into());
        }
        let inv12dy = 1.0 / (12.0 * self.dy);
        let inv2dy = 1.0 / (2.0 * self.dy);
        let invdy = 1.0 / self.dy;
        for i in 0..nx {
            for j in 2..ny - 2 {
                for k in 0..nz {
                    dst[[i, j, k]] = (8.0f64.mul_add(
                        -field[[i, j - 1, k]],
                        8.0f64.mul_add(field[[i, j + 1, k]], -field[[i, j + 2, k]]),
                    ) + field[[i, j - 2, k]])
                        * inv12dy;
                }
            }
        }
        for i in 0..nx {
            for k in 0..nz {
                if ny > 2 {
                    dst[[i, 1, k]] = (field[[i, 2, k]] - field[[i, 0, k]]) * inv2dy;
                    dst[[i, ny - 2, k]] = (field[[i, ny - 1, k]] - field[[i, ny - 3, k]]) * inv2dy;
                }
                dst[[i, 0, k]] = (field[[i, 1, k]] - field[[i, 0, k]]) * invdy;
                dst[[i, ny - 1, k]] = (field[[i, ny - 1, k]] - field[[i, ny - 2, k]]) * invdy;
            }
        }
        Ok(())
    }

    /// Apply ∂/∂z into a pre-allocated destination — zero heap allocation.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn apply_z_into(&self, field: ArrayView3<f64>, dst: &mut Array3<f64>) -> KwaversResult<()> {
        let (nx, ny, nz) = field.dim();
        if nz < 5 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 5,
                actual: nz,
                direction: "Z".to_owned(),
            }
            .into());
        }
        let inv12dz = 1.0 / (12.0 * self.dz);
        let inv2dz = 1.0 / (2.0 * self.dz);
        let invdz = 1.0 / self.dz;
        for i in 0..nx {
            for j in 0..ny {
                for k in 2..nz - 2 {
                    dst[[i, j, k]] = (8.0f64.mul_add(
                        -field[[i, j, k - 1]],
                        8.0f64.mul_add(field[[i, j, k + 1]], -field[[i, j, k + 2]]),
                    ) + field[[i, j, k - 2]])
                        * inv12dz;
                }
            }
        }
        for i in 0..nx {
            for j in 0..ny {
                if nz > 2 {
                    dst[[i, j, 1]] = (field[[i, j, 2]] - field[[i, j, 0]]) * inv2dz;
                    dst[[i, j, nz - 2]] = (field[[i, j, nz - 1]] - field[[i, j, nz - 3]]) * inv2dz;
                }
                dst[[i, j, 0]] = (field[[i, j, 1]] - field[[i, j, 0]]) * invdz;
                dst[[i, j, nz - 1]] = (field[[i, j, nz - 1]] - field[[i, j, nz - 2]]) * invdz;
            }
        }
        Ok(())
    }
}

impl DifferentialOperator for CentralDifference4 {
    fn apply_x(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();
        if nx < 5 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 5,
                actual: nx,
                direction: "X".to_owned(),
            }
            .into());
        }
        let mut result = Array3::zeros((nx, ny, nz));
        self.apply_x_into(field, &mut result)?;
        Ok(result)
    }

    fn apply_y(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();
        if ny < 5 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 5,
                actual: ny,
                direction: "Y".to_owned(),
            }
            .into());
        }
        let mut result = Array3::zeros((nx, ny, nz));
        self.apply_y_into(field, &mut result)?;
        Ok(result)
    }

    fn apply_z(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();
        if nz < 5 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 5,
                actual: nz,
                direction: "Z".to_owned(),
            }
            .into());
        }
        let mut result = Array3::zeros((nx, ny, nz));
        self.apply_z_into(field, &mut result)?;
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
