//! Fourth-order centered finite-difference stencils for scalar derivatives.

use ndarray::ArrayView3;

/// Centered finite-difference ∂f/∂x.
///
/// 4th-order stencil for `i ∈ [2, nx-3]`; 2nd-order one-sided otherwise.
#[inline]
#[must_use]
pub fn fd1_x(f: ArrayView3<f64>, i: usize, j: usize, k: usize, nx: usize, dx: f64) -> f64 {
    if nx <= 1 {
        return 0.0;
    }
    if i < 2 || i >= nx - 2 {
        if i == 0 {
            (f[[1, j, k]] - f[[0, j, k]]) / dx
        } else if i == nx - 1 {
            (f[[nx - 1, j, k]] - f[[nx - 2, j, k]]) / dx
        } else {
            (f[[i + 1, j, k]] - f[[i - 1, j, k]]) / (2.0 * dx)
        }
    } else {
        (8.0f64.mul_add(
            -f[[i - 1, j, k]],
            8.0f64.mul_add(f[[i + 1, j, k]], -f[[i + 2, j, k]]),
        ) + f[[i - 2, j, k]])
            / (12.0 * dx)
    }
}

/// Centered finite-difference ∂f/∂y.
///
/// 4th-order stencil for `j ∈ [2, ny-3]`; 2nd-order one-sided otherwise.
#[inline]
#[must_use]
pub fn fd1_y(f: ArrayView3<f64>, i: usize, j: usize, k: usize, ny: usize, dy: f64) -> f64 {
    if ny <= 1 {
        return 0.0;
    }
    if j < 2 || j >= ny - 2 {
        if j == 0 {
            (f[[i, 1, k]] - f[[i, 0, k]]) / dy
        } else if j == ny - 1 {
            (f[[i, ny - 1, k]] - f[[i, ny - 2, k]]) / dy
        } else {
            (f[[i, j + 1, k]] - f[[i, j - 1, k]]) / (2.0 * dy)
        }
    } else {
        (8.0f64.mul_add(
            -f[[i, j - 1, k]],
            8.0f64.mul_add(f[[i, j + 1, k]], -f[[i, j + 2, k]]),
        ) + f[[i, j - 2, k]])
            / (12.0 * dy)
    }
}

/// Centered finite-difference ∂f/∂z.
///
/// 4th-order stencil for `k ∈ [2, nz-3]`; 2nd-order one-sided otherwise.
#[inline]
#[must_use]
pub fn fd1_z(f: ArrayView3<f64>, i: usize, j: usize, k: usize, nz: usize, dz: f64) -> f64 {
    if nz <= 1 {
        return 0.0;
    }
    if k < 2 || k >= nz - 2 {
        if k == 0 {
            (f[[i, j, 1]] - f[[i, j, 0]]) / dz
        } else if k == nz - 1 {
            (f[[i, j, nz - 1]] - f[[i, j, nz - 2]]) / dz
        } else {
            (f[[i, j, k + 1]] - f[[i, j, k - 1]]) / (2.0 * dz)
        }
    } else {
        (8.0f64.mul_add(
            -f[[i, j, k - 1]],
            8.0f64.mul_add(f[[i, j, k + 1]], -f[[i, j, k + 2]]),
        ) + f[[i, j, k - 2]])
            / (12.0 * dz)
    }
}
