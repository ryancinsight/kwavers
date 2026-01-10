//! Gradient operations module

use super::coefficients::{FDCoefficients, SpatialOrder};
use crate::domain::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::{Array3, ArrayView3};
use num_traits::Float;

/// Compute the gradient of a 3D field
pub fn gradient<T>(
    field: &ArrayView3<T>,
    grid: &Grid,
    order: SpatialOrder,
) -> KwaversResult<(Array3<T>, Array3<T>, Array3<T>)>
where
    T: Float + Clone + Send + Sync,
{
    let shape = field.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);

    // Validate grid compatibility
    if (nx, ny, nz) != (grid.nx, grid.ny, grid.nz) {
        return Err(crate::domain::core::error::KwaversError::Grid(
            crate::domain::core::error::GridError::DimensionMismatch {
                expected: format!("({}, {}, {})", grid.nx, grid.ny, grid.nz),
                actual: format!("({}, {}, {})", nx, ny, nz),
            },
        ));
    }

    let mut grad_x = Array3::<T>::zeros((nx, ny, nz));
    let mut grad_y = Array3::<T>::zeros((nx, ny, nz));
    let mut grad_z = Array3::<T>::zeros((nx, ny, nz));

    let coeffs = FDCoefficients::first_derivative::<T>(order);
    let stencil_radius = coeffs.len();

    // X-direction gradient
    let dx_inv = T::one() / T::from(grid.dx).unwrap();
    for i in stencil_radius..nx - stencil_radius {
        for j in 0..ny {
            for k in 0..nz {
                let mut grad_val = T::zero();
                for (n, &coeff) in coeffs.iter().enumerate() {
                    let offset = n + 1;
                    grad_val =
                        grad_val + coeff * (field[[i + offset, j, k]] - field[[i - offset, j, k]]);
                }
                grad_x[[i, j, k]] = grad_val * dx_inv;
            }
        }
    }

    // Y-direction gradient
    let dy_inv = T::one() / T::from(grid.dy).unwrap();
    for i in 0..nx {
        for j in stencil_radius..ny - stencil_radius {
            for k in 0..nz {
                let mut grad_val = T::zero();
                for (n, &coeff) in coeffs.iter().enumerate() {
                    let offset = n + 1;
                    grad_val =
                        grad_val + coeff * (field[[i, j + offset, k]] - field[[i, j - offset, k]]);
                }
                grad_y[[i, j, k]] = grad_val * dy_inv;
            }
        }
    }

    // Z-direction gradient
    let dz_inv = T::one() / T::from(grid.dz).unwrap();
    for i in 0..nx {
        for j in 0..ny {
            for k in stencil_radius..nz - stencil_radius {
                let mut grad_val = T::zero();
                for (n, &coeff) in coeffs.iter().enumerate() {
                    let offset = n + 1;
                    grad_val =
                        grad_val + coeff * (field[[i, j, k + offset]] - field[[i, j, k - offset]]);
                }
                grad_z[[i, j, k]] = grad_val * dz_inv;
            }
        }
    }

    Ok((grad_x, grad_y, grad_z))
}
