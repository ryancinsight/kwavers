//! Shape validation and the centered stencil sum shared by the vector
//! operators (`curl`, `divergence`).

use crate::Grid;
use eunomia::FloatElement;
use kwavers_core::error::{GridError, KwaversError, KwaversResult};
use leto::ArrayView3;

/// The shared `[nx, ny, nz]` of a vector field's three components, checked
/// against the grid and against each other.
pub(super) fn validate_vector_field_shapes<T>(
    vx: &ArrayView3<T>,
    vy: &ArrayView3<T>,
    vz: &ArrayView3<T>,
    grid: &Grid,
) -> KwaversResult<[usize; 3]> {
    let shape = vx.shape();
    let dims = [shape[0], shape[1], shape[2]];
    let expected = [grid.nx, grid.ny, grid.nz];
    if dims != expected {
        return Err(KwaversError::Grid(GridError::DimensionMismatch {
            expected: format!("({}, {}, {})", grid.nx, grid.ny, grid.nz),
            actual: format!("({}, {}, {})", dims[0], dims[1], dims[2]),
        }));
    }

    if vy.shape() != shape || vz.shape() != shape {
        return Err(KwaversError::Grid(GridError::DimensionMismatch {
            expected: "Vector field components must have same dimensions".to_owned(),
            actual: format!(
                "vx: {:?}, vy: {:?}, vz: {:?}",
                vx.shape(),
                vy.shape(),
                vz.shape()
            ),
        }));
    }

    Ok(dims)
}

/// `Σₙ cₙ·δ(n + 1)`: a centered first-derivative stencil sum, with `δ(offset)`
/// the difference of the samples `offset` cells either side of the centre.
#[inline(always)]
pub(super) fn centered_first_derivative_sum<T, F>(coeffs: &[T], mut centered_delta: F) -> T
where
    T: FloatElement,
    F: FnMut(usize) -> T,
{
    let mut derivative = T::from_f64(0.0);
    for (n, &coeff) in coeffs.iter().enumerate() {
        let offset = n + 1;
        derivative += coeff * centered_delta(offset);
    }
    derivative
}
