//! Gradient operations module

use super::coefficients::FdAccuracyOrder;
use super::gradient_optimized::gradient_optimized;
use crate::Grid;
use eunomia::FloatElement;
use kwavers_core::error::KwaversResult;
use leto::{Array3, ArrayView3};

/// Compute centered derivatives on each axis. A component is zero where
/// its full stencil does not fit on that axis.
///
/// # Errors
/// Returns a dimension mismatch when the field shape differs from the grid.
pub fn gradient<T>(
    field: &ArrayView3<T>,
    grid: &Grid,
    order: FdAccuracyOrder,
) -> KwaversResult<(Array3<T>, Array3<T>, Array3<T>)>
where
    T: FloatElement + Clone + Send + Sync + Default,
{
    gradient_optimized(field, grid, order, None)
}
