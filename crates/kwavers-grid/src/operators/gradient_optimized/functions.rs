//! Gradient computation functions.

use super::super::coefficients::{FDCoefficients, FdAccuracyOrder};
use super::cache::GradientCache;
use super::operator::BoundaryStrategy;
use crate::compat::leto::{Array3, ArrayView3};
use crate::Grid;
use kwavers_core::error::KwaversResult;
use leto::Array3 as LetoArray3;
use eunomia::FloatElement;

/// Optimized gradient computation with caching and parallelization
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
pub fn gradient_optimized<T>(
    field: &ArrayView3<T>,
    grid: &Grid,
    order: FdAccuracyOrder,
    cache: Option<&GradientCache<T>>,
) -> KwaversResult<(Array3<T>, Array3<T>, Array3<T>)>
where
    T: Float + Clone + Send + Sync + Default,
{
    let shape = field.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);

    if (nx, ny, nz) != (grid.nx, grid.ny, grid.nz) {
        return Err(kwavers_core::error::KwaversError::Grid(
            kwavers_core::error::GridError::DimensionMismatch {
                expected: format!("({}, {}, {})", grid.nx, grid.ny, grid.nz),
                actual: format!("({}, {}, {})", nx, ny, nz),
            },
        ));
    }

    let mut grad_x = Array3::<T>::zeros([nx, ny, nz]);
    let mut grad_y = Array3::<T>::zeros([nx, ny, nz]);
    let mut grad_z = Array3::<T>::zeros([nx, ny, nz]);

    let coeffs = if let Some(cache) = cache {
        cache.get_coefficients(order)
    } else {
        FDCoefficients::first_derivative::<T>(order)
    };

    let stencil_radius = coeffs.len();
    let (dx_inv, dy_inv, dz_inv) = if let Some(cache) = cache {
        (
            cache.spacing_inverses.0,
            cache.spacing_inverses.1,
            cache.spacing_inverses.2,
        )
    } else {
        (
            T::one() / T::from(grid.dx).unwrap(),
            T::one() / T::from(grid.dy).unwrap(),
            T::one() / T::from(grid.dz).unwrap(),
        )
    };

    for i in stencil_radius..nx - stencil_radius {
        for j in 0..ny {
            for k in 0..nz {
                let mut grad_val = T::zero();
                for (n, coeff) in coeffs.iter().enumerate() {
                    let offset = n + 1;
                    grad_val =
                        grad_val + *coeff * (field[[i + offset, j, k]] - field[[i - offset, j, k]]);
                }
                grad_x[[i, j, k]] = grad_val * dx_inv;
            }
        }
    }

    for i in 0..nx {
        for j in stencil_radius..ny - stencil_radius {
            for k in 0..nz {
                let mut grad_val = T::zero();
                for (n, coeff) in coeffs.iter().enumerate() {
                    let offset = n + 1;
                    grad_val =
                        grad_val + *coeff * (field[[i, j + offset, k]] - field[[i, j - offset, k]]);
                }
                grad_y[[i, j, k]] = grad_val * dy_inv;
            }
        }
    }

    for i in 0..nx {
        for j in 0..ny {
            for k in stencil_radius..nz - stencil_radius {
                let mut grad_val = T::zero();
                for (n, coeff) in coeffs.iter().enumerate() {
                    let offset = n + 1;
                    grad_val =
                        grad_val + *coeff * (field[[i, j, k + offset]] - field[[i, j, k - offset]]);
                }
                grad_z[[i, j, k]] = grad_val * dz_inv;
            }
        }
    }

    Ok((grad_x, grad_y, grad_z))
}

/// Optimized gradient computation with boundary handling
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn gradient_with_boundaries<T>(
    field: &ArrayView3<T>,
    grid: &Grid,
    order: FdAccuracyOrder,
) -> KwaversResult<(Array3<T>, Array3<T>, Array3<T>)>
where
    T: Float + Clone + Send + Sync + Default,
{
    gradient_with_strategy(field, grid, order, BoundaryStrategy::ZeroPadding)
}
/// Gradient with strategy.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
pub(super) fn gradient_with_strategy<T>(
    field: &ArrayView3<T>,
    grid: &Grid,
    order: FdAccuracyOrder,
    boundary_strategy: BoundaryStrategy,
) -> KwaversResult<(Array3<T>, Array3<T>, Array3<T>)>
where
    T: Float + Clone + Send + Sync + Default,
{
    let shape = field.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);

    if (nx, ny, nz) != (grid.nx, grid.ny, grid.nz) {
        return Err(kwavers_core::error::KwaversError::Grid(
            kwavers_core::error::GridError::DimensionMismatch {
                expected: format!("({}, {}, {})", grid.nx, grid.ny, grid.nz),
                actual: format!("({}, {}, {})", nx, ny, nz),
            },
        ));
    }

    let mut grad_x = Array3::<T>::zeros([nx, ny, nz]);
    let mut grad_y = Array3::<T>::zeros([nx, ny, nz]);
    let mut grad_z = Array3::<T>::zeros([nx, ny, nz]);

    let coeffs = FDCoefficients::first_derivative::<T>(order);
    let _stencil_radius = coeffs.len();
    let dx_inv = T::one() / T::from(grid.dx).unwrap();
    let dy_inv = T::one() / T::from(grid.dy).unwrap();
    let dz_inv = T::one() / T::from(grid.dz).unwrap();

    let map_index = |idx: isize, len: usize| -> Option<usize> {
        if (0..(len as isize)).contains(&idx) {
            return Some(idx as usize);
        }
        match boundary_strategy {
            BoundaryStrategy::ZeroPadding => None,
            BoundaryStrategy::Mirror => {
                if len == 0 {
                    None
                } else {
                    let last = (len - 1) as isize;
                    let mirrored = if idx < 0 {
                        (-idx).min(last)
                    } else {
                        (2 * last - idx).max(0)
                    };
                    Some(mirrored as usize)
                }
            }
            BoundaryStrategy::Periodic => {
                if len == 0 {
                    None
                } else {
                    let m = len as isize;
                    let wrapped = ((idx % m) + m) % m;
                    Some(wrapped as usize)
                }
            }
            BoundaryStrategy::Extrapolate => {
                if len == 0 {
                    None
                } else if idx < 0 {
                    Some(0)
                } else {
                    Some(len - 1)
                }
            }
        }
    };

    let get = |ii: isize, jj: isize, kk: isize| -> T {
        let Some(iu) = map_index(ii, nx) else {
            return T::zero();
        };
        let Some(ju) = map_index(jj, ny) else {
            return T::zero();
        };
        let Some(ku) = map_index(kk, nz) else {
            return T::zero();
        };
        field[[iu, ju, ku]]
    };

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let (ii, jj, kk) = (i as isize, j as isize, k as isize);

                let mut gx = T::zero();
                for (n, coeff) in coeffs.iter().enumerate() {
                    let offset = (n + 1) as isize;
                    gx = gx + *coeff * (get(ii + offset, jj, kk) - get(ii - offset, jj, kk));
                }
                grad_x[[i, j, k]] = gx * dx_inv;

                let mut gy = T::zero();
                for (n, coeff) in coeffs.iter().enumerate() {
                    let offset = (n + 1) as isize;
                    gy = gy + *coeff * (get(ii, jj + offset, kk) - get(ii, jj - offset, kk));
                }
                grad_y[[i, j, k]] = gy * dy_inv;

                let mut gz = T::zero();
                for (n, coeff) in coeffs.iter().enumerate() {
                    let offset = (n + 1) as isize;
                    gz = gz + *coeff * (get(ii, jj, kk + offset) - get(ii, jj, kk - offset));
                }
                grad_z[[i, j, k]] = gz * dz_inv;
            }
        }
    }

    Ok((grad_x, grad_y, grad_z))
}

/// Optimized gradient computation for leto fields.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn gradient_optimized_leto<T>(
    field: &LetoArray3<T>,
    grid: &Grid,
    order: FdAccuracyOrder,
    cache: Option<&GradientCache<T>>,
) -> KwaversResult<(LetoArray3<T>, LetoArray3<T>, LetoArray3<T>)>
where
    T: Float + Clone + Send + Sync + Default,
{
    let field_view = field.view();
    gradient_optimized(&field_view, grid, order, cache)
}

/// Gradient computation with boundary handling for leto fields.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn gradient_with_boundaries_leto<T>(
    field: &LetoArray3<T>,
    grid: &Grid,
    order: FdAccuracyOrder,
) -> KwaversResult<(LetoArray3<T>, LetoArray3<T>, LetoArray3<T>)>
where
    T: Float + Clone + Send + Sync + Default,
{
    let field_view = field.view();
    gradient_with_boundaries(&field_view, grid, order)
}
