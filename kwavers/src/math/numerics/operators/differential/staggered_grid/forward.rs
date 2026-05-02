//! Forward difference methods for `StaggeredGridOperator`.
//!
//! SRP: changes when the forward stencil or allocation strategy changes.

use super::operator::StaggeredGridOperator;
use crate::core::error::{KwaversResult, NumericalError};
use ndarray::{s, Array3, ArrayView3, Zip};

impl StaggeredGridOperator {
    /// Apply forward difference in X into a pre-allocated buffer.
    ///
    /// Zero heap allocation. `dst` must have shape `(nx-1, ny, nz)`.
    /// `dst[i,j,k] = (field[i+1,j,k] − field[i,j,k]) / Δx`
    pub fn apply_forward_x_into(
        &self,
        field: ArrayView3<f64>,
        dst: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = field.dim();
        if nx < 2 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 2,
                actual: nx,
                direction: "X".to_string(),
            }
            .into());
        }
        debug_assert_eq!(
            dst.dim(),
            (nx - 1, ny, nz),
            "apply_forward_x_into: dst shape {dst:?} does not match expected ({}, {ny}, {nz})",
            nx - 1
        );
        let dx = self.dx;
        Zip::from(dst)
            .and(field.slice(s![1.., .., ..]))
            .and(field.slice(s![..nx - 1, .., ..]))
            .for_each(|r, &hi, &lo| *r = (hi - lo) / dx);
        Ok(())
    }

    /// Apply forward difference in Y into a pre-allocated buffer.
    ///
    /// Zero heap allocation. `dst` must have shape `(nx, ny-1, nz)`.
    /// `dst[i,j,k] = (field[i,j+1,k] − field[i,j,k]) / Δy`
    pub fn apply_forward_y_into(
        &self,
        field: ArrayView3<f64>,
        dst: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = field.dim();
        if ny < 2 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 2,
                actual: ny,
                direction: "Y".to_string(),
            }
            .into());
        }
        debug_assert_eq!(
            dst.dim(),
            (nx, ny - 1, nz),
            "apply_forward_y_into: dst shape {dst:?} does not match expected ({nx}, {}, {nz})",
            ny - 1
        );
        let dy = self.dy;
        Zip::from(dst)
            .and(field.slice(s![.., 1.., ..]))
            .and(field.slice(s![.., ..ny - 1, ..]))
            .for_each(|r, &hi, &lo| *r = (hi - lo) / dy);
        Ok(())
    }

    /// Apply forward difference in Z into a pre-allocated buffer.
    ///
    /// Zero heap allocation. `dst` must have shape `(nx, ny, nz-1)`.
    /// `dst[i,j,k] = (field[i,j,k+1] − field[i,j,k]) / Δz`
    pub fn apply_forward_z_into(
        &self,
        field: ArrayView3<f64>,
        dst: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = field.dim();
        if nz < 2 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 2,
                actual: nz,
                direction: "Z".to_string(),
            }
            .into());
        }
        debug_assert_eq!(
            dst.dim(),
            (nx, ny, nz - 1),
            "apply_forward_z_into: dst shape {dst:?} does not match expected ({nx}, {ny}, {})",
            nz - 1
        );
        let dz = self.dz;
        Zip::from(dst)
            .and(field.slice(s![.., .., 1..]))
            .and(field.slice(s![.., .., ..nz - 1]))
            .for_each(|r, &hi, &lo| *r = (hi - lo) / dz);
        Ok(())
    }

    /// Apply forward difference in X, allocating the result.
    ///
    /// `∂u/∂x|_{i+1/2,j,k} ≈ (u[i+1,j,k] - u[i,j,k]) / Δx`
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
        self.apply_forward_x_into(field, &mut result)?;
        Ok(result)
    }

    pub fn apply_forward_y(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
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
        self.apply_forward_y_into(field, &mut result)?;
        Ok(result)
    }

    pub fn apply_forward_z(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
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
        self.apply_forward_z_into(field, &mut result)?;
        Ok(result)
    }
}
