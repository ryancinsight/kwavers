//! Forward difference methods for `StaggeredGridOperator`.
//!
//! SRP: changes when the forward stencil or allocation strategy changes.

use super::operator::StaggeredGridOperator;
use crate::numerics::operators::differential::traversal;
use kwavers_core::error::{KwaversResult, NumericalError};
use leto::{Array3, ArrayView3};
use leto_ops::zip2_mut_with;

impl StaggeredGridOperator {
    /// Apply forward difference in X into a pre-allocated buffer.
    ///
    /// Zero heap allocation. `dst` must have shape `(nx-1, ny, nz)`.
    /// `dst[i,j,k] = (field[i+1,j,k] − field[i,j,k]) / Δx`
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
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
        if field.is_c_contiguous() {
            if let Some(field_values) = field.as_slice() {
                if traversal::try_fill_standard_layout(dst, |i, j, k| {
                    (field_values[traversal::row_major_index(i + 1, j, k, ny, nz)]
                        - field_values[traversal::row_major_index(i, j, k, ny, nz)])
                        / dx
                }) {
                    return Ok(());
                }
            }
        }
        let mut dst_slice = dst
            .slice_mut(&[(0, nx - 1, 1), (0, ny, 1), (0, nz, 1)])
            .unwrap();
        let field_hi = field.slice(&[(1, nx, 1), (0, ny, 1), (0, nz, 1)]).unwrap();
        let field_lo = field
            .slice(&[(0, nx - 1, 1), (0, ny, 1), (0, nz, 1)])
            .unwrap();
        zip2_mut_with(&mut dst_slice, &field_hi, &field_lo, |r, &hi, &lo| {
            *r = (hi - lo) / dx
        })
        .unwrap();
        Ok(())
    }

    /// Apply forward difference in Y into a pre-allocated buffer.
    ///
    /// Zero heap allocation. `dst` must have shape `(nx, ny-1, nz)`.
    /// `dst[i,j,k] = (field[i,j+1,k] − field[i,j,k]) / Δy`
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
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
        if field.is_c_contiguous() {
            if let Some(field_values) = field.as_slice() {
                if traversal::try_fill_standard_layout(dst, |i, j, k| {
                    (field_values[traversal::row_major_index(i, j + 1, k, ny, nz)]
                        - field_values[traversal::row_major_index(i, j, k, ny, nz)])
                        / dy
                }) {
                    return Ok(());
                }
            }
        }
        let mut dst_slice = dst
            .slice_mut(&[(0, nx, 1), (0, ny - 1, 1), (0, nz, 1)])
            .unwrap();
        let field_hi = field.slice(&[(0, nx, 1), (1, ny, 1), (0, nz, 1)]).unwrap();
        let field_lo = field
            .slice(&[(0, nx, 1), (0, ny - 1, 1), (0, nz, 1)])
            .unwrap();
        zip2_mut_with(&mut dst_slice, &field_hi, &field_lo, |r, &hi, &lo| {
            *r = (hi - lo) / dy
        })
        .unwrap();
        Ok(())
    }

    /// Apply forward difference in Z into a pre-allocated buffer.
    ///
    /// Zero heap allocation. `dst` must have shape `(nx, ny, nz-1)`.
    /// `dst[i,j,k] = (field[i,j,k+1] − field[i,j,k]) / Δz`
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
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
        if field.is_c_contiguous() {
            if let Some(field_values) = field.as_slice() {
                if traversal::try_fill_standard_layout(dst, |i, j, k| {
                    (field_values[traversal::row_major_index(i, j, k + 1, ny, nz)]
                        - field_values[traversal::row_major_index(i, j, k, ny, nz)])
                        / dz
                }) {
                    return Ok(());
                }
            }
        }
        let mut dst_slice = dst
            .slice_mut(&[(0, nx, 1), (0, ny, 1), (0, nz - 1, 1)])
            .unwrap();
        let field_hi = field.slice(&[(0, nx, 1), (0, ny, 1), (1, nz, 1)]).unwrap();
        let field_lo = field
            .slice(&[(0, nx, 1), (0, ny, 1), (0, nz - 1, 1)])
            .unwrap();
        zip2_mut_with(&mut dst_slice, &field_hi, &field_lo, |r, &hi, &lo| {
            *r = (hi - lo) / dz
        })
        .unwrap();
        Ok(())
    }

    /// Apply forward difference in X, allocating the result.
    ///
    /// `∂u/∂x|_{i+1/2,j,k} ≈ (u[i+1,j,k] - u[i,j,k]) / Δx`
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
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
    /// - Propagates any `KwaversError` returned by called functions.
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
    /// - Propagates any `KwaversError` returned by called functions.
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
