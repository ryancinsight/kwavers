//! Gradient operations module

use super::coefficients::FdAccuracyOrder;
use super::gradient_optimized::gradient_optimized;
use crate::Grid;
use eunomia::FloatElement;
use kwavers_core::error::KwaversResult;
use leto::{Array3, ArrayView3};

/// Compute the gradient of a 3D field without a coefficient cache.
///
/// The kernel is shared with [`gradient_optimized`]; this entry point keeps
/// the uncached convenience contract while avoiding a second implementation
/// of the centered finite-difference stencil.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
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
