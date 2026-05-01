//! Backward-difference kernels for the staggered-grid operator.
//!
//! Computes derivatives at cell centers (i) from cell edges (i-1, i) — the
//! path used by FDTD velocity → pressure updates. The first interior plane
//! (`i=0` / `j=0` / `k=0`) falls back to a forward difference because no
//! `i−1` point exists; this preserves second-order accuracy in the bulk and
//! gives a one-sided difference at the boundary.

use ndarray::{s, Array3, ArrayView3, Zip};

use super::StaggeredGridOperator;
use crate::core::error::{KwaversResult, NumericalError};

impl StaggeredGridOperator {
    /// Apply backward difference in X direction into a pre-allocated destination buffer.
    ///
    /// Zero heap allocation. `dst` must have shape `(nx, ny, nz)`.
    ///
    /// ## Algorithm
    ///
    /// Interior `i ∈ [1, nx)`: backward difference
    /// ```text
    /// dst[i,j,k] = (field[i,j,k] − field[i−1,j,k]) / Δx
    /// ```
    /// Boundary `i = 0`: forward difference (no `i−1` point)
    /// ```text
    /// dst[0,j,k] = (field[1,j,k] − field[0,j,k]) / Δx
    /// ```
    ///
    /// Both passes use `Zip` slice-pairs for LLVM auto-vectorization.
    ///
    /// # Errors
    ///
    /// Returns `InsufficientGridPoints` if `nx < 2`.
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
                direction: "X".to_string(),
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
        // Interior: dst[1..nx] = (field[1..] − field[..nx−1]) / Δx
        Zip::from(dst.slice_mut(s![1.., .., ..]))
            .and(field.slice(s![1.., .., ..]))
            .and(field.slice(s![..nx - 1, .., ..]))
            .par_for_each(|r, &hi, &lo| *r = (hi - lo) / dx);
        // Boundary i=0: forward difference
        Zip::from(dst.slice_mut(s![0, .., ..]))
            .and(field.slice(s![1, .., ..]))
            .and(field.slice(s![0, .., ..]))
            .for_each(|r, &hi, &lo| *r = (hi - lo) / dx);
        Ok(())
    }

    /// Apply backward difference in Y direction into a pre-allocated destination buffer.
    ///
    /// Zero heap allocation. `dst` must have shape `(nx, ny, nz)`.
    ///
    /// Interior `j ∈ [1, ny)`:
    /// ```text
    /// dst[i,j,k] = (field[i,j,k] − field[i,j−1,k]) / Δy
    /// ```
    /// Boundary `j = 0`: forward difference.
    ///
    /// # Errors
    ///
    /// Returns `InsufficientGridPoints` if `ny < 2`.
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
                direction: "Y".to_string(),
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

    /// Apply backward difference in Z direction into a pre-allocated destination buffer.
    ///
    /// Zero heap allocation. `dst` must have shape `(nx, ny, nz)`.
    ///
    /// Interior `k ∈ [1, nz)`:
    /// ```text
    /// dst[i,j,k] = (field[i,j,k] − field[i,j,k−1]) / Δz
    /// ```
    /// Boundary `k = 0`: forward difference.
    ///
    /// # Errors
    ///
    /// Returns `InsufficientGridPoints` if `nz < 2`.
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
                direction: "Z".to_string(),
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

    /// Apply backward difference in X direction.
    ///
    /// Computes derivative at cell centers (i) from cell edges (i-1, i).
    /// Used for velocity → pressure updates in FDTD.
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
        self.apply_backward_x_into(field, &mut result)?;
        Ok(result)
    }

    /// Apply backward difference in Y direction.
    ///
    /// Computes derivative at cell centers from cell edges in Y direction.
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
        self.apply_backward_y_into(field, &mut result)?;
        Ok(result)
    }

    /// Apply backward difference in Z direction.
    ///
    /// Computes derivative at cell centers from cell edges in Z direction.
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
        self.apply_backward_z_into(field, &mut result)?;
        Ok(result)
    }
}
