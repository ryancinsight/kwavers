//! Forward difference methods for `StaggeredGridOperator`.
//!
//! SRP: changes when the forward stencil or allocation strategy changes.

use super::operator::StaggeredGridOperator;
use kwavers_core::error::{KwaversResult, NumericalError};
use leto::{Array3, ArrayView3};

impl StaggeredGridOperator {
    /// Apply forward difference in X into a pre-allocated buffer.
    ///
    /// Zero heap allocation. `dst` must have shape `(nx-1, ny, nz)`.
    /// `dst[i,j,k] = (field[i+1,j,k] − field[i,j,k]) / Δx`
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    pub fn apply_forward_x_into(
        &self,
        field: ArrayView3<f64>,
        dst: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let [nx, ny, nz] = field.shape();
        if nx < 2 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 2,
                actual: nx,
                direction: "X".to_owned(),
            }
            .into());
        }
        debug_assert_eq!(
            dst.shape(),
            [nx - 1, ny, nz],
            "apply_forward_x_into: dst shape {dst:?} does not match expected ({}, {ny}, {nz})",
            nx - 1
        );
        let dx = self.dx;
        for i in 0..nx - 1 {
            for j in 0..ny {
                for k in 0..nz {
                    dst[[i, j, k]] = (field[[i + 1, j, k]] - field[[i, j, k]]) / dx;
                }
            }
        }
        Ok(())
    }

    /// Apply forward difference in Y into a pre-allocated buffer.
    ///
    /// Zero heap allocation. `dst` must have shape `(nx, ny-1, nz)`.
    /// `dst[i,j,k] = (field[i,j+1,k] − field[i,j,k]) / Δy`
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    pub fn apply_forward_y_into(
        &self,
        field: ArrayView3<f64>,
        dst: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let [nx, ny, nz] = field.shape();
        if ny < 2 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 2,
                actual: ny,
                direction: "Y".to_owned(),
            }
            .into());
        }
        debug_assert_eq!(
            dst.shape(),
            [nx, ny - 1, nz],
            "apply_forward_y_into: dst shape {dst:?} does not match expected ({nx}, {}, {nz})",
            ny - 1
        );
        let dy = self.dy;
        for i in 0..nx {
            for j in 0..ny - 1 {
                for k in 0..nz {
                    dst[[i, j, k]] = (field[[i, j + 1, k]] - field[[i, j, k]]) / dy;
                }
            }
        }
        Ok(())
    }

    /// Apply forward difference in Z into a pre-allocated buffer.
    ///
    /// Zero heap allocation. `dst` must have shape `(nx, ny, nz-1)`.
    /// `dst[i,j,k] = (field[i,j,k+1] − field[i,j,k]) / Δz`
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    pub fn apply_forward_z_into(
        &self,
        field: ArrayView3<f64>,
        dst: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let [nx, ny, nz] = field.shape();
        if nz < 2 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 2,
                actual: nz,
                direction: "Z".to_owned(),
            }
            .into());
        }
        debug_assert_eq!(
            dst.shape(),
            [nx, ny, nz - 1],
            "apply_forward_z_into: dst shape {dst:?} does not match expected ({nx}, {ny}, {})",
            nz - 1
        );
        let dz = self.dz;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz - 1 {
                    dst[[i, j, k]] = (field[[i, j, k + 1]] - field[[i, j, k]]) / dz;
                }
            }
        }
        Ok(())
    }

    /// Apply forward difference in X, allocating the result.
    ///
    /// `∂u/∂x|_{i+1/2,j,k} ≈ (u[i+1,j,k] - u[i,j,k]) / Δx`
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn apply_forward_x(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let [nx, ny, nz] = field.shape();
        if nx < 2 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 2,
                actual: nx,
                direction: "X".to_owned(),
            }
            .into());
        }
        let mut result = Array3::zeros([nx - 1, ny, nz]);
        self.apply_forward_x_into(field, &mut result)?;
        Ok(result)
    }
    /// Apply forward y.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn apply_forward_y(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let [nx, ny, nz] = field.shape();
        if ny < 2 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 2,
                actual: ny,
                direction: "Y".to_owned(),
            }
            .into());
        }
        let mut result = Array3::zeros([nx, ny - 1, nz]);
        self.apply_forward_y_into(field, &mut result)?;
        Ok(result)
    }
    /// Apply forward z.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn apply_forward_z(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let [nx, ny, nz] = field.shape();
        if nz < 2 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 2,
                actual: nz,
                direction: "Z".to_owned(),
            }
            .into());
        }
        let mut result = Array3::zeros([nx, ny, nz - 1]);
        self.apply_forward_z_into(field, &mut result)?;
        Ok(result)
    }
}
