//! Kernel and windowed operations over fields

use ndarray::Array3;
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
    T: Clone + Send + Sync,
    K: Clone + Send + Sync + PartialEq + Default,
    U: Clone + Send + Sync + Default + Add<Output = U>,
{
    let (nx, ny, nz) = field.dim();
    let (kx, ky, kz) = kernel.dim();
    let (hx, hy, hz) = (kx / 2, ky / 2, kz / 2);

    // Pre-compute non-zero kernel elements for sparse optimization
    let non_zero_kernel: Vec<((usize, usize, usize), &K)> = kernel
        .indexed_iter()
        .filter(|(_, kval)| **kval != K::default())
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
pub fn apply_kernel_parallel<T, K, U>(
    field: &Array3<T>,
    kernel: &Array3<K>,
    combine: impl Fn(&T, &K) -> U + Sync + Send,
) -> Array3<U>
where
    T: Clone + Send + Sync,
    K: Clone + Send + Sync + PartialEq + Default,
    U: Clone + Send + Sync + Default + Add<Output = U>,
{
    let (nx, ny, nz) = field.dim();
    let (kx, ky, kz) = kernel.dim();
    let (hx, hy, hz) = (kx / 2, ky / 2, kz / 2);

    let non_zero_kernel: Vec<((usize, usize, usize), K)> = kernel
        .indexed_iter()
        .filter(|(_, kval)| **kval != K::default())
        .map(|(idx, kval)| (idx, kval.clone()))
        .collect();

    let indices: Vec<(usize, usize, usize)> = (0..nx)
        .flat_map(|i| (0..ny).flat_map(move |j| (0..nz).map(move |k| (i, j, k))))
        .collect();

    let flat_results: Vec<U> = indices
        .par_iter()
        .map(|&(i, j, k)| {
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
    T: Clone,
    U: Clone + Default,
    F: Fn(&Array3<T>) -> U,
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

        if x_end > x_start && y_end > y_start && z_end > z_start {
            let window = field.slice(ndarray::s![x_start..x_end, y_start..y_end, z_start..z_end]);
            operation(&window.to_owned())
        } else {
            U::default()
        }
    })
}
