//! Dispatch enum unifying 2nd / 4th / 6th-order central-difference operators.
//!
//! Each variant wraps a concrete `CentralDifferenceN` operator and forwards
//! single-axis derivative calls (`apply_x/y/z` and zero-allocation
//! `apply_x/y/z_into`) plus a packed `gradient` helper.

use ndarray::{Array3, ArrayView3};

use crate::core::error::{KwaversError, KwaversResult};
use crate::math::numerics::operators::{
    CentralDifference2, CentralDifference4, CentralDifference6, DifferentialOperator,
};

#[derive(Debug, Clone)]
pub(crate) enum CentralDifferenceOperator {
    Order2(CentralDifference2),
    Order4(CentralDifference4),
    Order6(CentralDifference6),
}

impl CentralDifferenceOperator {
    /// New.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(crate) fn new(order: usize, dx: f64, dy: f64, dz: f64) -> KwaversResult<Self> {
        match order {
            2 => Ok(Self::Order2(CentralDifference2::new(dx, dy, dz)?)),
            4 => Ok(Self::Order4(CentralDifference4::new(dx, dy, dz)?)),
            6 => Ok(Self::Order6(CentralDifference6::new(dx, dy, dz)?)),
            _ => Err(KwaversError::InvalidInput(format!(
                "spatial_order must be 2, 4, or 6, got {order}"
            ))),
        }
    }
    /// Apply x.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(crate) fn apply_x(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        match self {
            Self::Order2(op) => op.apply_x(field),
            Self::Order4(op) => op.apply_x(field),
            Self::Order6(op) => op.apply_x(field),
        }
    }
    /// Apply y.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(crate) fn apply_y(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        match self {
            Self::Order2(op) => op.apply_y(field),
            Self::Order4(op) => op.apply_y(field),
            Self::Order6(op) => op.apply_y(field),
        }
    }
    /// Apply z.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(crate) fn apply_z(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        match self {
            Self::Order2(op) => op.apply_z(field),
            Self::Order4(op) => op.apply_z(field),
            Self::Order6(op) => op.apply_z(field),
        }
    }

    /// Apply X-derivative in-place into a pre-allocated destination buffer.
    ///
    /// Zero heap allocation for all orders: O2 via `CentralDifference2::apply_x_into`,
    /// O4 via `CentralDifference4::apply_x_into`, O6 via `CentralDifference6::apply_x_into`.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(crate) fn apply_x_into(
        &self,
        field: ArrayView3<f64>,
        dst: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        match self {
            Self::Order2(op) => op.apply_x_into(field, dst),
            Self::Order4(op) => op.apply_x_into(field, dst),
            Self::Order6(op) => op.apply_x_into(field, dst),
        }
    }

    /// Apply Y-derivative in-place into a pre-allocated destination buffer.
    ///
    /// Zero heap allocation for all orders.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(crate) fn apply_y_into(
        &self,
        field: ArrayView3<f64>,
        dst: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        match self {
            Self::Order2(op) => op.apply_y_into(field, dst),
            Self::Order4(op) => op.apply_y_into(field, dst),
            Self::Order6(op) => op.apply_y_into(field, dst),
        }
    }

    /// Apply Z-derivative in-place into a pre-allocated destination buffer.
    ///
    /// Zero heap allocation for all orders.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(crate) fn apply_z_into(
        &self,
        field: ArrayView3<f64>,
        dst: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        match self {
            Self::Order2(op) => op.apply_z_into(field, dst),
            Self::Order4(op) => op.apply_z_into(field, dst),
            Self::Order6(op) => op.apply_z_into(field, dst),
        }
    }
    /// Gradient.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(crate) fn gradient(
        &self,
        field: ArrayView3<f64>,
    ) -> KwaversResult<(Array3<f64>, Array3<f64>, Array3<f64>)> {
        Ok((
            self.apply_x(field)?,
            self.apply_y(field)?,
            self.apply_z(field)?,
        ))
    }
}
