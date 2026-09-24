//! Differential operators module for discretized grids
//!
//! This module provides modular differential operators:
//! - Information Expert: Each operator type owns its implementation
//! - Single Responsibility: Separate modules for each operator category
//! - High Cohesion: Related operations grouped together
use crate::Grid;
use eunomia::FloatElement;
use kwavers_core::error::{GridError, KwaversError, KwaversResult};
use leto::ArrayView3;

pub mod coefficients;
pub mod curl;
pub mod divergence;
pub mod gradient;
pub mod gradient_optimized;
pub mod laplacian;

// Re-export main types
pub use coefficients::{FDCoefficients, FdAccuracyOrder};
pub use curl::curl;
pub use divergence::divergence;
pub use gradient::gradient;
pub use gradient_optimized::{GradientCache, GradientOperator, GradientOperatorBuilder};
pub use laplacian::laplacian;

pub(crate) fn validate_vector_field_shapes<T>(
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

#[inline(always)]
pub(crate) fn centered_first_derivative_sum<T, F>(coeffs: &[T], mut centered_delta: F) -> T
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
