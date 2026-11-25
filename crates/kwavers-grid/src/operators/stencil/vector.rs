//! Shared traversal for centered vector derivatives with a complete-stencil interior.
use super::{inverse_spacing, validate_vector_field_shapes};
use crate::{operators::FdAccuracyOrder, Grid};
use eunomia::FloatElement;
use kwavers_core::error::{GridError, KwaversError, KwaversResult};
use leto::{Array3, ArrayView3};
mod interior;

/// Cartesian directions of both the vector components and the grid axes.
#[derive(Clone, Copy)]
pub(in crate::operators) enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    const fn index(self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }
}

/// Selects the entries of the vector field's Jacobian needed by an operator.
/// The arguments to `partial` are component then differentiation axis.
pub(in crate::operators) trait VectorDerivative<const N: usize> {
    fn evaluate<T: FloatElement>(partial: impl Fn(Axis, Axis) -> T) -> [T; N];
}

pub(in crate::operators) fn owned<
    O: VectorDerivative<N>,
    T: FloatElement + Default,
    const N: usize,
>(
    fields: [&ArrayView3<T>; 3],
    grid: &Grid,
    order: FdAccuracyOrder,
) -> KwaversResult<[Array3<T>; N]> {
    let [x, y, z] = fields;
    let shape = validate_vector_field_shapes(x, y, z, grid)?;
    let mut output = std::array::from_fn(|_| Array3::zeros(shape));
    let slices = output
        .each_mut()
        .map(|v| v.as_slice_mut().expect("fresh array is contiguous"));
    apply::<O, T, N, false>(fields, grid, order, slices);
    Ok(output)
}

pub(in crate::operators) fn into<O: VectorDerivative<N>, T: FloatElement, const N: usize>(
    fields: [&ArrayView3<T>; 3],
    grid: &Grid,
    order: FdAccuracyOrder,
    output: [&mut [T]; N],
) -> KwaversResult<()> {
    let [x, y, z] = fields;
    validate_vector_field_shapes(x, y, z, grid)?;
    let size = x.size();
    for (component, dst) in output.iter().enumerate() {
        if dst.len() != size {
            return Err(KwaversError::Grid(GridError::DimensionMismatch {
                expected: format!("{size} elements for output component {component}"),
                actual: dst.len().to_string(),
            }));
        }
    }
    apply::<O, T, N, true>(fields, grid, order, output);
    Ok(())
}

fn apply<O: VectorDerivative<N>, T: FloatElement, const N: usize, const CLEAR_HALO: bool>(
    fields: [&ArrayView3<T>; 3],
    grid: &Grid,
    order: FdAccuracyOrder,
    output: [&mut [T]; N],
) {
    let shape @ [_, ny, nz] = fields[0].shape();
    let weights = order.first_weights();
    let coeffs = &weights[..order.radius()];
    let scale = inverse_spacing(grid);
    if let [Some(x), Some(y), Some(z)] = fields.map(|f| f.as_slice()) {
        let data = [x, y, z];
        interior::write::<O, T, N, CLEAR_HALO>(shape, coeffs, scale, output, |c, [i, j, k]| {
            data[c][(i * ny + j) * nz + k]
        });
    } else {
        interior::write::<O, T, N, CLEAR_HALO>(shape, coeffs, scale, output, |c, p| fields[c][p]);
    }
}
