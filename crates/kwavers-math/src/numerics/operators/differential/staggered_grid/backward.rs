//! Backward difference methods for `StaggeredGridOperator`.
//!
//! SRP: changes when the backward stencil or boundary treatment changes.

use super::operator::StaggeredGridOperator;
use kwavers_core::error::{KwaversResult, NumericalError};
use ndarray::{s, Array3, ArrayView3, Zip};

impl StaggeredGridOperator {
    /// Apply backward difference in X into a pre-allocated buffer.
    ///
    /// Zero heap allocation. `dst` must have shape `(nx, ny, nz)`.
    /// At i=0: forward difference (no i-1 point).
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    pub fn apply_backward_x_into(
        &self,
        field: ArrayView3<f64>,
        dst: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = field.dim();
        if nx < 2 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 2,
                actual: nx,
                direction: "X".to_owned(),
            }
            .into());
        }
        debug_assert_eq!(
            dst.dim(),
            (nx, ny, nz),
            "apply_backward_x_into: dst shape {:?} must be ({nx}, {ny}, {nz})",
            dst.dim()
        );
        let dx = self.dx;
        Zip::from(dst.slice_mut(s![1.., .., ..]))
            .and(field.slice(s![1.., .., ..]))
            .and(field.slice(s![..nx - 1, .., ..]))
            .par_for_each(|r, &hi, &lo| *r = (hi - lo) / dx);
        Zip::from(dst.slice_mut(s![0, .., ..]))
            .and(field.slice(s![1, .., ..]))
            .and(field.slice(s![0, .., ..]))
            .for_each(|r, &hi, &lo| *r = (hi - lo) / dx);
        Ok(())
    }

    /// Apply backward difference in Y into a pre-allocated buffer.
    ///
    /// Zero heap allocation. `dst` must have shape `(nx, ny, nz)`.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    pub fn apply_backward_y_into(
        &self,
        field: ArrayView3<f64>,
        dst: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = field.dim();
        if ny < 2 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 2,
                actual: ny,
                direction: "Y".to_owned(),
            }
            .into());
        }
        debug_assert_eq!(
            dst.dim(),
            (nx, ny, nz),
            "apply_backward_y_into: dst shape {:?} must be ({nx}, {ny}, {nz})",
            dst.dim()
        );
        let dy = self.dy;
        Zip::from(dst.slice_mut(s![.., 1.., ..]))
            .and(field.slice(s![.., 1.., ..]))
            .and(field.slice(s![.., ..ny - 1, ..]))
            .par_for_each(|r, &hi, &lo| *r = (hi - lo) / dy);
        Zip::from(dst.slice_mut(s![.., 0, ..]))
            .and(field.slice(s![.., 1, ..]))
            .and(field.slice(s![.., 0, ..]))
            .for_each(|r, &hi, &lo| *r = (hi - lo) / dy);
        Ok(())
    }

    /// Apply backward difference in Z into a pre-allocated buffer.
    ///
    /// Zero heap allocation. `dst` must have shape `(nx, ny, nz)`.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    pub fn apply_backward_z_into(
        &self,
        field: ArrayView3<f64>,
        dst: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = field.dim();
        if nz < 2 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 2,
                actual: nz,
                direction: "Z".to_owned(),
            }
            .into());
        }
        debug_assert_eq!(
            dst.dim(),
            (nx, ny, nz),
            "apply_backward_z_into: dst shape {:?} must be ({nx}, {ny}, {nz})",
            dst.dim()
        );
        let dz = self.dz;
        Zip::from(dst.slice_mut(s![.., .., 1..]))
            .and(field.slice(s![.., .., 1..]))
            .and(field.slice(s![.., .., ..nz - 1]))
            .par_for_each(|r, &hi, &lo| *r = (hi - lo) / dz);
        Zip::from(dst.slice_mut(s![.., .., 0]))
            .and(field.slice(s![.., .., 1]))
            .and(field.slice(s![.., .., 0]))
            .for_each(|r, &hi, &lo| *r = (hi - lo) / dz);
        Ok(())
    }

    /// Apply backward difference in X, allocating the result.
    ///
    /// `∂u/∂x|_{i,j,k} ≈ (u[i,j,k] - u[i-1,j,k]) / Δx`.
    /// At i=0, uses forward difference.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn apply_backward_x(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();
        if nx < 2 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 2,
                actual: nx,
                direction: "X".to_owned(),
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
        for j in 0..ny {
            for k in 0..nz {
                if nx > 1 {
                    result[[0, j, k]] = (field[[1, j, k]] - field[[0, j, k]]) / self.dx;
                }
            }
        }
        Ok(result)
    }
    /// Apply backward y.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn apply_backward_y(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();
        if ny < 2 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 2,
                actual: ny,
                direction: "Y".to_owned(),
            }
            .into());
        }
        let mut result = Array3::zeros((nx, ny, nz));
        for i in 0..nx {
            for j in 1..ny {
                for k in 0..nz {
                    result[[i, j, k]] = (field[[i, j, k]] - field[[i, j - 1, k]]) / self.dy;
                }
            }
        }
        for i in 0..nx {
            for k in 0..nz {
                if ny > 1 {
                    result[[i, 0, k]] = (field[[i, 1, k]] - field[[i, 0, k]]) / self.dy;
                }
            }
        }
        Ok(result)
    }
    /// Apply backward z.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn apply_backward_z(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();
        if nz < 2 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 2,
                actual: nz,
                direction: "Z".to_owned(),
            }
            .into());
        }
        let mut result = Array3::zeros((nx, ny, nz));
        for i in 0..nx {
            for j in 0..ny {
                for k in 1..nz {
                    result[[i, j, k]] = (field[[i, j, k]] - field[[i, j, k - 1]]) / self.dz;
                }
            }
        }
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
