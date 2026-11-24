//! Public gradient entry points and the shared grid/precision dispatch.

use super::{kernel, BoundaryStrategy, GradientCache};
use crate::{
    operators::{
        stencil::{inverse_spacing, validate_field_shape},
        FdAccuracyOrder,
    },
    Grid,
};
use eunomia::FloatElement;
use kwavers_core::error::KwaversResult;
use leto::{Array3, ArrayView3};

/// Compute centered derivatives, leaving zero in each component's incomplete halo.
/// A supplied cache can be reused across grids with different spacings.
///
/// # Errors
/// Returns a dimension mismatch when the field shape differs from the grid.
pub fn gradient_optimized<T: FloatElement + Default>(
    field: &ArrayView3<T>,
    grid: &Grid,
    order: FdAccuracyOrder,
    cache: Option<&GradientCache<T>>,
) -> KwaversResult<(Array3<T>, Array3<T>, Array3<T>)> {
    compute_gradient(field, grid, order, cache, None, None)
}

/// Compute the gradient with zero-valued samples outside the grid.
///
/// # Errors
/// Returns a dimension mismatch when the field shape differs from the grid.
pub fn gradient_with_boundaries<T: FloatElement + Default>(
    field: &ArrayView3<T>,
    grid: &Grid,
    order: FdAccuracyOrder,
) -> KwaversResult<(Array3<T>, Array3<T>, Array3<T>)> {
    compute_gradient(
        field,
        grid,
        order,
        None,
        Some(BoundaryStrategy::ZeroPadding),
        None,
    )
}

pub(super) fn compute_gradient<T: FloatElement + Default>(
    field: &ArrayView3<T>,
    grid: &Grid,
    order: FdAccuracyOrder,
    cache: Option<&GradientCache<T>>,
    boundary: Option<BoundaryStrategy>,
    parallel_rows: Option<usize>,
) -> KwaversResult<(Array3<T>, Array3<T>, Array3<T>)> {
    let shape = validate_field_shape(field, grid)?;
    let [mut x, mut y, mut z] = std::array::from_fn(|_| Array3::zeros(shape));
    // Owned arrays constructed above have standard contiguous layout.
    fill_gradient(
        field,
        grid,
        order,
        cache,
        boundary,
        parallel_rows,
        [
            x.as_slice_mut().expect("contiguous gradient output"),
            y.as_slice_mut().expect("contiguous gradient output"),
            z.as_slice_mut().expect("contiguous gradient output"),
        ],
    );
    Ok((x, y, z))
}

pub(super) fn fill_gradient<T: FloatElement>(
    field: &ArrayView3<T>,
    grid: &Grid,
    order: FdAccuracyOrder,
    cache: Option<&GradientCache<T>>,
    boundary: Option<BoundaryStrategy>,
    parallel_rows: Option<usize>,
    outputs: [&mut [T]; 3],
) {
    let inline;
    let weights = match cache {
        Some(cache) => cache.get_coefficients(order),
        None => {
            inline = order.first_weights();
            &inline[..order.radius()]
        }
    };
    let scale = inverse_spacing(grid);
    kernel::fill(field, weights, scale, boundary, parallel_rows, outputs);
}
