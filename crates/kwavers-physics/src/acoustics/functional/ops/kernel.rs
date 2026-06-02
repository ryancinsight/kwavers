//! Kernel and windowed operations over fields

use ndarray::{Array3, ArrayView3};
use rayon::prelude::*;
use std::ops::Add;

/// Compose multiple field operations efficiently
pub fn compose_operations<T, F1, F2, U, V>(field: &Array3<T>, op1: F1, op2: F2) -> Array3<V>
where
    F1: Fn(&Array3<T>) -> Array3<U>,
    F2: Fn(&Array3<U>) -> Array3<V>,
{
    op2(&op1(field))
}

/// Kernel operation with sparse kernel support
pub fn apply_kernel<T, K, U>(
    field: &Array3<T>,
    kernel: &Array3<K>,
    combine: impl Fn(&T, &K) -> U + Sync + Send,
) -> Array3<U>
where
    T: Sync,
    K: Sync + PartialEq + Default,
    U: Default + Add<Output = U>,
{
    let (nx, ny, nz) = field.dim();
    let (kx, ky, kz) = kernel.dim();
    let (hx, hy, hz) = (kx / 2, ky / 2, kz / 2);

    // Pre-compute non-zero kernel references once. This keeps sparse kernel
    // evaluation borrowed and avoids requiring coefficient cloning.
    let default_kernel = K::default();
    let non_zero_kernel: Vec<((usize, usize, usize), &K)> = kernel
        .indexed_iter()
        .filter(|(_, kval)| *kval != &default_kernel)
        .collect();

    Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
        non_zero_kernel
            .iter()
            .filter_map(|((ki, kj, kk), kval)| {
                let fi = (i + ki).wrapping_sub(hx);
                let fj = (j + kj).wrapping_sub(hy);
                let fk = (k + kk).wrapping_sub(hz);

                if fi < nx && fj < ny && fk < nz {
                    Some(combine(&field[[fi, fj, fk]], kval))
                } else {
                    None
                }
            })
            .fold(U::default(), |acc, val| acc + val)
    })
}

/// Parallel kernel application for large fields
/// # Panics
/// - Panics if `Shape mismatch in parallel kernel application`.
///
pub fn apply_kernel_parallel<T, K, U>(
    field: &Array3<T>,
    kernel: &Array3<K>,
    combine: impl Fn(&T, &K) -> U + Sync + Send,
) -> Array3<U>
where
    T: Sync,
    K: Sync + PartialEq + Default,
    U: Send + Default + Add<Output = U>,
{
    let (nx, ny, nz) = field.dim();
    let (kx, ky, kz) = kernel.dim();
    let (hx, hy, hz) = (kx / 2, ky / 2, kz / 2);

    let default_kernel = K::default();
    let non_zero_kernel: Vec<((usize, usize, usize), &K)> = kernel
        .indexed_iter()
        .filter(|(_, kval)| *kval != &default_kernel)
        .collect();

    let total = nx * ny * nz;
    let yz = ny * nz;
    let flat_results: Vec<U> = (0..total)
        .into_par_iter()
        .map(|flat| {
            let i = flat / yz;
            let rem = flat % yz;
            let j = rem / nz;
            let k = rem % nz;
            non_zero_kernel
                .iter()
                .filter_map(|((ki, kj, kk), kval)| {
                    let fi = (i + ki).wrapping_sub(hx);
                    let fj = (j + kj).wrapping_sub(hy);
                    let fk = (k + kk).wrapping_sub(hz);

                    if fi < nx && fj < ny && fk < nz {
                        Some(combine(&field[[fi, fj, fk]], kval))
                    } else {
                        None
                    }
                })
                .fold(U::default(), |acc, val| acc + val)
        })
        .collect();

    Array3::from_shape_vec((nx, ny, nz), flat_results)
        .expect("Shape mismatch in parallel kernel application")
}

/// Windowed operation over a field
pub fn windowed_operation<T, U, F>(
    field: &Array3<T>,
    window_size: (usize, usize, usize),
    operation: F,
) -> Array3<U>
where
    F: Fn(ArrayView3<'_, T>) -> U,
{
    let (nx, ny, nz) = field.dim();
    let (wx, wy, wz) = window_size;
    let (hwx, hwy, hwz) = (wx / 2, wy / 2, wz / 2);

    Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
        let x_start = i.saturating_sub(hwx);
        let x_end = (i + hwx + 1).min(nx);
        let y_start = j.saturating_sub(hwy);
        let y_end = (j + hwy + 1).min(ny);
        let z_start = k.saturating_sub(hwz);
        let z_end = (k + hwz + 1).min(nz);

        let window = field.slice(ndarray::s![x_start..x_end, y_start..y_end, z_start..z_end]);
        operation(window)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Default, PartialEq, Eq)]
    struct NonCloneScalar(i32);

    impl Add for NonCloneScalar {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Self(self.0 + rhs.0)
        }
    }

    /// compose_operations(op1, op2) applies op1 then op2 sequentially.
    ///
    /// op1: ×2, op2: +1 → compose(v) = 2v+1. At uniform v=3: result = 7.
    #[test]
    fn compose_operations_applies_in_sequence() {
        let field: Array3<f64> = Array3::from_elem((3, 3, 3), 3.0);
        let result = compose_operations(&field, |f| f.mapv(|x| x * 2.0), |f| f.mapv(|x| x + 1.0));
        for &v in result.iter() {
            assert!(
                (v - 7.0).abs() < 1e-14,
                "compose(×2,+1) at 3.0 must give 7.0 (got {v})"
            );
        }
    }

    /// apply_kernel_parallel produces identical results to apply_kernel.
    ///
    /// Identity-weight kernel (centre=1) → output equals input.
    #[test]
    fn apply_kernel_parallel_matches_sequential_result() {
        let field: Array3<f64> = Array3::from_shape_fn((5, 5, 5), |(i, j, k)| (i + j + k) as f64);
        let mut kernel = Array3::<f64>::zeros((3, 3, 3));
        kernel[[1, 1, 1]] = 1.0; // identity kernel

        let seq = apply_kernel(&field, &kernel, |f: &f64, k: &f64| f * k);
        let par = apply_kernel_parallel(&field, &kernel, |f: &f64, k: &f64| f * k);

        for ((i, j, k), &sv) in seq.indexed_iter() {
            let pv = par[[i, j, k]];
            assert!(
                (pv - sv).abs() < 1e-14,
                "[{i},{j},{k}]: sequential={sv:.4}, parallel={pv:.4}"
            );
        }
    }

    /// Borrowed sparse kernel evaluation must not require `Clone` on field,
    /// kernel, or output values.
    ///
    /// Center-only kernel: `u_out(i,j,k)=2u(i,j,k)`.
    #[test]
    fn apply_kernel_accepts_non_clone_scalars() {
        let field =
            Array3::from_shape_fn((3, 3, 3), |(i, j, k)| NonCloneScalar((i + j + k) as i32));
        let kernel = Array3::from_shape_fn((3, 3, 3), |(i, j, k)| {
            if (i, j, k) == (1, 1, 1) {
                NonCloneScalar(2)
            } else {
                NonCloneScalar(0)
            }
        });

        let seq = apply_kernel(&field, &kernel, |f, k| NonCloneScalar(f.0 * k.0));
        let par = apply_kernel_parallel(&field, &kernel, |f, k| NonCloneScalar(f.0 * k.0));

        assert_eq!(seq[[2, 1, 0]].0, 6);
        assert_eq!(par[[2, 1, 0]].0, 6);
    }

    /// Windowed operations receive borrowed views rather than owned windows.
    ///
    /// For `u(i,j,k)=i+j+k`, the centered 3³ window around `(1,1,1)` has
    /// mean 3 by symmetry.
    #[test]
    fn windowed_operation_passes_borrowed_views() {
        let field =
            Array3::from_shape_fn((3, 3, 3), |(i, j, k)| NonCloneScalar((i + j + k) as i32));

        let result = windowed_operation(&field, (3, 3, 3), |window| {
            let sum: i32 = window.iter().map(|value| value.0).sum();
            sum / window.len() as i32
        });

        assert_eq!(result[[1, 1, 1]], 3);
        assert_eq!(result[[0, 0, 0]], 1);
    }
}
