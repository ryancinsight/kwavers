//! Backward difference methods for `StaggeredGridOperator`.
//!
//! SRP: changes when the backward stencil or boundary treatment changes.

use super::operator::StaggeredGridOperator;
use crate::numerics::operators::differential::traversal;
use kwavers_core::error::{KwaversResult, NumericalError};
use leto::{Array3, ArrayView3};
use leto_ops::zip2_mut_with;

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
            [nx, ny, nz],
            "apply_backward_x_into: dst shape {:?} must be ({nx}, {ny}, {nz})",
            dst.shape()
        );
        let dx = self.dx;
        if field.is_c_contiguous() {
            if let Some(field_values) = field.as_slice() {
                if traversal::try_fill_standard_layout(dst, |i, j, k| {
                    let center = traversal::row_major_index(i, j, k, ny, nz);
                    if i == 0 {
                        (field_values[traversal::row_major_index(1, j, k, ny, nz)]
                            - field_values[center])
                            / dx
                    } else {
                        (field_values[center]
                            - field_values[traversal::row_major_index(i - 1, j, k, ny, nz)])
                            / dx
                    }
                }) {
                    return Ok(());
                }
            }
        }
        let mut dst_slice = dst
            .slice_mut(&[(1, nx, 1), (0, ny, 1), (0, nz, 1)])
            .unwrap();
        let field_hi = field.slice(&[(1, nx, 1), (0, ny, 1), (0, nz, 1)]).unwrap();
        let field_lo = field
            .slice(&[(0, nx - 1, 1), (0, ny, 1), (0, nz, 1)])
            .unwrap();
        zip2_mut_with(&mut dst_slice, &field_hi, &field_lo, |r, &hi, &lo| {
            *r = (hi - lo) / dx
        })
        .unwrap();
        let mut dst_slice = dst.slice_mut(&[(0, 1, 1), (0, ny, 1), (0, nz, 1)]).unwrap();
        let field_hi = field.slice(&[(1, 2, 1), (0, ny, 1), (0, nz, 1)]).unwrap();
        let field_lo = field.slice(&[(0, 1, 1), (0, ny, 1), (0, nz, 1)]).unwrap();
        zip2_mut_with(&mut dst_slice, &field_hi, &field_lo, |r, &hi, &lo| {
            *r = (hi - lo) / dx
        })
        .unwrap();
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
            [nx, ny, nz],
            "apply_backward_y_into: dst shape {:?} must be ({nx}, {ny}, {nz})",
            dst.shape()
        );
        let dy = self.dy;
        if field.is_c_contiguous() {
            if let Some(field_values) = field.as_slice() {
                if traversal::try_fill_standard_layout(dst, |i, j, k| {
                    let center = traversal::row_major_index(i, j, k, ny, nz);
                    if j == 0 {
                        (field_values[traversal::row_major_index(i, 1, k, ny, nz)]
                            - field_values[center])
                            / dy
                    } else {
                        (field_values[center]
                            - field_values[traversal::row_major_index(i, j - 1, k, ny, nz)])
                            / dy
                    }
                }) {
                    return Ok(());
                }
            }
        }
        let mut dst_slice = dst
            .slice_mut(&[(0, nx, 1), (1, ny, 1), (0, nz, 1)])
            .unwrap();
        let field_hi = field.slice(&[(0, nx, 1), (1, ny, 1), (0, nz, 1)]).unwrap();
        let field_lo = field
            .slice(&[(0, nx, 1), (0, ny - 1, 1), (0, nz, 1)])
            .unwrap();
        zip2_mut_with(&mut dst_slice, &field_hi, &field_lo, |r, &hi, &lo| {
            *r = (hi - lo) / dy
        })
        .unwrap();
        let mut dst_slice = dst.slice_mut(&[(0, nx, 1), (0, 1, 1), (0, nz, 1)]).unwrap();
        let field_hi = field.slice(&[(0, nx, 1), (1, 2, 1), (0, nz, 1)]).unwrap();
        let field_lo = field.slice(&[(0, nx, 1), (0, 1, 1), (0, nz, 1)]).unwrap();
        zip2_mut_with(&mut dst_slice, &field_hi, &field_lo, |r, &hi, &lo| {
            *r = (hi - lo) / dy
        })
        .unwrap();
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
            [nx, ny, nz],
            "apply_backward_z_into: dst shape {:?} must be ({nx}, {ny}, {nz})",
            dst.shape()
        );
        let dz = self.dz;
        if field.is_c_contiguous() {
            if let Some(field_values) = field.as_slice() {
                if traversal::try_fill_standard_layout(dst, |i, j, k| {
                    let center = traversal::row_major_index(i, j, k, ny, nz);
                    if k == 0 {
                        (field_values[traversal::row_major_index(i, j, 1, ny, nz)]
                            - field_values[center])
                            / dz
                    } else {
                        (field_values[center]
                            - field_values[traversal::row_major_index(i, j, k - 1, ny, nz)])
                            / dz
                    }
                }) {
                    return Ok(());
                }
            }
        }
        let mut dst_slice = dst
            .slice_mut(&[(0, nx, 1), (0, ny, 1), (1, nz, 1)])
            .unwrap();
        let field_hi = field.slice(&[(0, nx, 1), (0, ny, 1), (1, nz, 1)]).unwrap();
        let field_lo = field
            .slice(&[(0, nx, 1), (0, ny, 1), (0, nz - 1, 1)])
            .unwrap();
        zip2_mut_with(&mut dst_slice, &field_hi, &field_lo, |r, &hi, &lo| {
            *r = (hi - lo) / dz
        })
        .unwrap();
        let mut dst_slice = dst.slice_mut(&[(0, nx, 1), (0, ny, 1), (0, 1, 1)]).unwrap();
        let field_hi = field.slice(&[(0, nx, 1), (0, ny, 1), (1, 2, 1)]).unwrap();
        let field_lo = field.slice(&[(0, nx, 1), (0, ny, 1), (0, 1, 1)]).unwrap();
        zip2_mut_with(&mut dst_slice, &field_hi, &field_lo, |r, &hi, &lo| {
            *r = (hi - lo) / dz
        })
        .unwrap();
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
        let [nx, ny, nz] = field.shape();
        if nx < 2 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 2,
                actual: nx,
                direction: "X".to_owned(),
            }
            .into());
        }
        let mut result = Array3::zeros([nx, ny, nz]);
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
        let [nx, ny, nz] = field.shape();
        if ny < 2 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 2,
                actual: ny,
                direction: "Y".to_owned(),
            }
            .into());
        }
        let mut result = Array3::zeros([nx, ny, nz]);
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
        let [nx, ny, nz] = field.shape();
        if nz < 2 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 2,
                actual: nz,
                direction: "Z".to_owned(),
            }
            .into());
        }
        let mut result = Array3::zeros([nx, ny, nz]);
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
