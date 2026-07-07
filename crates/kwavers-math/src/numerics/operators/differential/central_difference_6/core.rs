//! # Sixth-Order Central Difference Operator
//!
//! Implements the sixth-order accurate central difference scheme
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
//! ## References
//!
//! - Fornberg, B. (1988). "Generation of finite difference formulas on arbitrarily
//!   spaced grids." *Mathematics of Computation*, 51(184), 699-706.
//!   DOI: 10.1090/S0025-5718-1988-0935077-0

use kwavers_core::error::{KwaversResult, NumericalError};
use leto::{Array3, ArrayView3};

use super::super::DifferentialOperator;

/// Sixth-order central difference operator.
///
/// Provides highest-accuracy central difference for wave propagation:
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
    /// Create a new sixth-order central difference operator.
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

    /// Apply ∂/∂x into a pre-allocated destination — zero heap allocation.
    ///
    /// Interior stencil (O(Δx⁶)):
    /// ```text
    /// dst[i] = (−f[i−3] + 9f[i−2] − 45f[i−1] + 45f[i+1] − 9f[i+2] + f[i+3]) / (60 Δx)
    /// ```
    /// Near-boundary: O(Δx⁴) at i=2/n−3, O(Δx²) at i=1/n−2, O(Δx) at i=0/n−1.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn apply_x_into(&self, field: ArrayView3<f64>, dst: &mut Array3<f64>) -> KwaversResult<()> {
        let [nx, ny, nz] = field.shape();
        if nx < 7 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 7,
                actual: nx,
                direction: "X".to_owned(),
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
                    dst[[i, j, k]] = (9.0f64.mul_add(
                        -field[[i + 2, j, k]],
                        45.0f64.mul_add(
                            field[[i + 1, j, k]],
                            45.0f64.mul_add(
                                -field[[i - 1, j, k]],
                                9.0f64.mul_add(field[[i - 2, j, k]], -field[[i - 3, j, k]]),
                            ),
                        ),
                    ) + field[[i + 3, j, k]])
                        * inv60dx;
                }
            }
        }
        for j in 0..ny {
            for k in 0..nz {
                dst[[2, j, k]] = (8.0f64.mul_add(
                    -field[[1, j, k]],
                    8.0f64.mul_add(field[[3, j, k]], -field[[4, j, k]]),
                ) + field[[0, j, k]])
                    * inv12dx;
                dst[[nx - 3, j, k]] = (8.0f64.mul_add(
                    -field[[nx - 4, j, k]],
                    8.0f64.mul_add(field[[nx - 2, j, k]], -field[[nx - 1, j, k]]),
                ) + field[[nx - 5, j, k]])
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn apply_y_into(&self, field: ArrayView3<f64>, dst: &mut Array3<f64>) -> KwaversResult<()> {
        let [nx, ny, nz] = field.shape();
        if ny < 7 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 7,
                actual: ny,
                direction: "Y".to_owned(),
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
                    dst[[i, j, k]] = (9.0f64.mul_add(
                        -field[[i, j + 2, k]],
                        45.0f64.mul_add(
                            field[[i, j + 1, k]],
                            45.0f64.mul_add(
                                -field[[i, j - 1, k]],
                                9.0f64.mul_add(field[[i, j - 2, k]], -field[[i, j - 3, k]]),
                            ),
                        ),
                    ) + field[[i, j + 3, k]])
                        * inv60dy;
                }
            }
        }
        for i in 0..nx {
            for k in 0..nz {
                dst[[i, 2, k]] = (8.0f64.mul_add(
                    -field[[i, 1, k]],
                    8.0f64.mul_add(field[[i, 3, k]], -field[[i, 4, k]]),
                ) + field[[i, 0, k]])
                    * inv12dy;
                dst[[i, ny - 3, k]] = (8.0f64.mul_add(
                    -field[[i, ny - 4, k]],
                    8.0f64.mul_add(field[[i, ny - 2, k]], -field[[i, ny - 1, k]]),
                ) + field[[i, ny - 5, k]])
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn apply_z_into(&self, field: ArrayView3<f64>, dst: &mut Array3<f64>) -> KwaversResult<()> {
        let [nx, ny, nz] = field.shape();
        if nz < 7 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 7,
                actual: nz,
                direction: "Z".to_owned(),
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
                    dst[[i, j, k]] = (9.0f64.mul_add(
                        -field[[i, j, k + 2]],
                        45.0f64.mul_add(
                            field[[i, j, k + 1]],
                            45.0f64.mul_add(
                                -field[[i, j, k - 1]],
                                9.0f64.mul_add(field[[i, j, k - 2]], -field[[i, j, k - 3]]),
                            ),
                        ),
                    ) + field[[i, j, k + 3]])
                        * inv60dz;
                }
            }
        }
        for i in 0..nx {
            for j in 0..ny {
                dst[[i, j, 2]] = (8.0f64.mul_add(
                    -field[[i, j, 1]],
                    8.0f64.mul_add(field[[i, j, 3]], -field[[i, j, 4]]),
                ) + field[[i, j, 0]])
                    * inv12dz;
                dst[[i, j, nz - 3]] = (8.0f64.mul_add(
                    -field[[i, j, nz - 4]],
                    8.0f64.mul_add(field[[i, j, nz - 2]], -field[[i, j, nz - 1]]),
                ) + field[[i, j, nz - 5]])
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
        let [nx, ny, nz] = field.shape();
        if nx < 7 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 7,
                actual: nx,
                direction: "X".to_owned(),
            }
            .into());
        }
        let mut result = Array3::zeros([nx, ny, nz]);
        self.apply_x_into(field, &mut result)?;
        Ok(result)
    }

    fn apply_y(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let [nx, ny, nz] = field.shape();
        if ny < 7 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 7,
                actual: ny,
                direction: "Y".to_owned(),
            }
            .into());
        }
        let mut result = Array3::zeros([nx, ny, nz]);
        self.apply_y_into(field, &mut result)?;
        Ok(result)
    }

    fn apply_z(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let [nx, ny, nz] = field.shape();
        if nz < 7 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 7,
                actual: nz,
                direction: "Z".to_owned(),
            }
            .into());
        }
        let mut result = Array3::zeros([nx, ny, nz]);
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
        true
    }
}
