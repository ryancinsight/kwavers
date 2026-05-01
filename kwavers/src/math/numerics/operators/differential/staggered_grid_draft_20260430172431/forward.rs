//! Forward-difference kernels for the staggered-grid operator.
//!
//! Computes derivatives at cell edges (i+1/2) from cell centers (i, i+1) —
//! the path used by FDTD pressure → velocity updates. Both zero-allocation
//! `_into` variants and allocating wrappers are provided.

use ndarray::{s, Array3, ArrayView3, Zip};

use super::StaggeredGridOperator;
use crate::core::error::{KwaversResult, NumericalError};

impl StaggeredGridOperator {
    /// Apply forward difference in X direction into a pre-allocated destination buffer.
    ///
    /// Zero heap allocation. `dst` must have shape `(nx-1, ny, nz)`.
    ///
    /// Uses a vectorizable slice-pair pattern:
    /// ```text
    /// dst[i,j,k] = (field[i+1,j,k] − field[i,j,k]) / Δx
    /// ```
    /// The `Zip::from(dst).and(field[1..]).and(field[..n-1])` structure exposes the
    /// element-wise independence to LLVM auto-vectorization.
    ///
    /// # Errors
    ///
    /// Returns `InsufficientGridPoints` if `nx < 2`.
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

    /// Apply forward difference in Y direction into a pre-allocated destination buffer.
    ///
    /// Zero heap allocation. `dst` must have shape `(nx, ny-1, nz)`.
    ///
    /// ```text
    /// dst[i,j,k] = (field[i,j+1,k] − field[i,j,k]) / Δy
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `InsufficientGridPoints` if `ny < 2`.
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

    /// Apply forward difference in Z direction into a pre-allocated destination buffer.
    ///
    /// Zero heap allocation. `dst` must have shape `(nx, ny, nz-1)`.
    ///
    /// ```text
    /// dst[i,j,k] = (field[i,j,k+1] − field[i,j,k]) / Δz
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `InsufficientGridPoints` if `nz < 2`.
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

    /// Apply forward difference in X direction.
    ///
    /// Computes derivative at cell edges (i+1/2) from cell centers (i, i+1).
    /// Used for pressure → velocity updates in FDTD.
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
        self.apply_forward_x_into(field, &mut result)?;
        Ok(result)
    }

    /// Apply forward difference in Y direction.
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

    /// Apply forward difference in Z direction.
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
