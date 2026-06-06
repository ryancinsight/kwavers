//! Fourth-order centered finite-difference stencils for scalar derivatives.
//!
//! The three public functions [`fd1_x`], [`fd1_y`], [`fd1_z`] are thin axis
//! wrappers over a single shared [`fd1_axis`] implementation: the boundary axis
//! is selected at runtime and the stencil reads are addressed via a mutated
//! index, so the arithmetic is bit-identical to per-axis literal indexing (the
//! `tests` module pins this against the original literal formulas).

use ndarray::ArrayView3;

/// Read `f` at `base` shifted by `off` along `axis` (caller guarantees the
/// resulting index is in bounds for the stencil branch in use).
#[inline]
fn shifted(f: ArrayView3<f64>, base: [usize; 3], axis: usize, off: isize) -> f64 {
    let mut idx = base;
    idx[axis] = (idx[axis] as isize + off) as usize;
    f[idx]
}

/// Centered finite-difference ∂f/∂(axis): 4th-order for the interior
/// `c ∈ [2, n-3]`, 2nd-order centered for `c ∈ {1, n-2}`, first-order one-sided
/// at the `c ∈ {0, n-1}` edges. Returns 0 for a singleton axis (`n <= 1`).
#[inline]
fn fd1_axis(f: ArrayView3<f64>, base: [usize; 3], axis: usize, n: usize, d: f64) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let c = base[axis];
    if c < 2 || c >= n - 2 {
        if c == 0 {
            (shifted(f, base, axis, 1) - shifted(f, base, axis, 0)) / d
        } else if c == n - 1 {
            (shifted(f, base, axis, 0) - shifted(f, base, axis, -1)) / d
        } else {
            (shifted(f, base, axis, 1) - shifted(f, base, axis, -1)) / (2.0 * d)
        }
    } else {
        (8.0f64.mul_add(
            -shifted(f, base, axis, -1),
            8.0f64.mul_add(shifted(f, base, axis, 1), -shifted(f, base, axis, 2)),
        ) + shifted(f, base, axis, -2))
            / (12.0 * d)
    }
}

/// Centered finite-difference ∂f/∂x (see [`fd1_axis`]).
#[inline]
#[must_use]
pub fn fd1_x(f: ArrayView3<f64>, i: usize, j: usize, k: usize, nx: usize, dx: f64) -> f64 {
    fd1_axis(f, [i, j, k], 0, nx, dx)
}

/// Centered finite-difference ∂f/∂y (see [`fd1_axis`]).
#[inline]
#[must_use]
pub fn fd1_y(f: ArrayView3<f64>, i: usize, j: usize, k: usize, ny: usize, dy: f64) -> f64 {
    fd1_axis(f, [i, j, k], 1, ny, dy)
}

/// Centered finite-difference ∂f/∂z (see [`fd1_axis`]).
#[inline]
#[must_use]
pub fn fd1_z(f: ArrayView3<f64>, i: usize, j: usize, k: usize, nz: usize, dz: f64) -> f64 {
    fd1_axis(f, [i, j, k], 2, nz, dz)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    // Reference implementations using the original per-axis *literal* indexing.
    // The axis-generic wrappers above must match these bit-for-bit.
    fn ref_x(f: ArrayView3<f64>, i: usize, j: usize, k: usize, nx: usize, dx: f64) -> f64 {
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
            (8.0f64.mul_add(-f[[i - 1, j, k]], 8.0f64.mul_add(f[[i + 1, j, k]], -f[[i + 2, j, k]]))
                + f[[i - 2, j, k]])
                / (12.0 * dx)
        }
    }
    fn ref_y(f: ArrayView3<f64>, i: usize, j: usize, k: usize, ny: usize, dy: f64) -> f64 {
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
            (8.0f64.mul_add(-f[[i, j - 1, k]], 8.0f64.mul_add(f[[i, j + 1, k]], -f[[i, j + 2, k]]))
                + f[[i, j - 2, k]])
                / (12.0 * dy)
        }
    }
    fn ref_z(f: ArrayView3<f64>, i: usize, j: usize, k: usize, nz: usize, dz: f64) -> f64 {
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
            (8.0f64.mul_add(-f[[i, j, k - 1]], 8.0f64.mul_add(f[[i, j, k + 1]], -f[[i, j, k + 2]]))
                + f[[i, j, k - 2]])
                / (12.0 * dz)
        }
    }

    #[test]
    fn axis_generic_matches_literal_indexing() {
        let (nx, ny, nz) = (7usize, 6usize, 5usize);
        let f = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
            (i * 31 + j * 17 + k * 7) as f64 * 0.0137 + 0.5
        });
        let (dx, dy, dz) = (1e-3, 2e-3, 3e-3);
        let v = f.view();
        // Sweep every index along each axis (covers edges, 2nd-order, interior).
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    assert_eq!(fd1_x(v, i, j, k, nx, dx), ref_x(v, i, j, k, nx, dx));
                    assert_eq!(fd1_y(v, i, j, k, ny, dy), ref_y(v, i, j, k, ny, dy));
                    assert_eq!(fd1_z(v, i, j, k, nz, dz), ref_z(v, i, j, k, nz, dz));
                }
            }
        }
    }

    #[test]
    fn singleton_axis_is_zero() {
        let f = Array3::from_shape_fn((1, 4, 4), |(i, j, k)| (i + j + k) as f64);
        let v = f.view();
        assert_eq!(fd1_x(v, 0, 1, 1, 1, 1e-3), 0.0);
    }

    #[test]
    fn fourth_order_exact_for_cubic() {
        // The 4th-order centered stencil differentiates cubics exactly.
        let n = 9usize;
        let dx = 0.1;
        let f = Array3::from_shape_fn((n, 1, 1), |(i, _, _)| {
            let x = i as f64 * dx;
            x * x * x // f = x³ → f' = 3x²
        });
        let v = f.view();
        for i in 2..n - 2 {
            let x = i as f64 * dx;
            assert!((fd1_x(v, i, 0, 0, n, dx) - 3.0 * x * x).abs() < 1e-9);
        }
    }
}
