//! Field Operations and Combinators
//!
//! This module provides enhanced field operations with iterator support,
//! better ergonomics, and optimizations for sparse operations.

use ndarray::Array3;
use rayon::prelude::*;
use std::ops::Add;

/// Field operations trait with iterator support
pub trait FieldOps {
    type Item;

    /// Map a function over all elements (now supports FnMut for stateful closures)
    fn map_field<F, U>(&self, f: F) -> Array3<U>
    where
        F: FnMut(&Self::Item) -> U;

    /// Filter and return iterator of matching indices (lazy evaluation)
    fn filter_indices<'a, F>(
        &'a self,
        predicate: F,
    ) -> impl Iterator<Item = (usize, usize, usize)> + 'a
    where
        F: Fn(&Self::Item) -> bool + 'a;

    /// Fold over the field with an accumulator
    fn fold_field<F, U>(&self, init: U, f: F) -> U
    where
        F: Fn(U, &Self::Item) -> U;

    /// Scan over the field, producing intermediate results
    fn scan_field<F, U>(&self, init: U, f: F) -> Array3<U>
    where
        F: Fn(&U, &Self::Item) -> U,
        U: Clone;

    /// Parallel map operation for better performance
    fn par_map_field<F, U>(&self, f: F) -> Array3<U>
    where
        F: Fn(&Self::Item) -> U + Sync + Send,
        U: Send + Sync,
        Self::Item: Sync;

    /// Find the first element matching a predicate
    fn find_element<F>(&self, predicate: F) -> Option<((usize, usize, usize), &Self::Item)>
    where
        F: Fn(&Self::Item) -> bool;

    /// Count elements matching a predicate
    fn count_matching<F>(&self, predicate: F) -> usize
    where
        F: Fn(&Self::Item) -> bool;

    /// Check if any element matches a predicate
    fn any<F>(&self, predicate: F) -> bool
    where
        F: Fn(&Self::Item) -> bool;

    /// Check if all elements match a predicate
    fn all<F>(&self, predicate: F) -> bool
    where
        F: Fn(&Self::Item) -> bool;
}

impl<T: Clone + Send + Sync> FieldOps for Array3<T> {
    type Item = T;

    fn map_field<F, U>(&self, mut f: F) -> Array3<U>
    where
        F: FnMut(&Self::Item) -> U,
    {
        let shape = self.dim();
        Array3::from_shape_fn(shape, |(i, j, k)| f(&self[[i, j, k]]))
    }

    fn filter_indices<'a, F>(
        &'a self,
        predicate: F,
    ) -> impl Iterator<Item = (usize, usize, usize)> + 'a
    where
        F: Fn(&Self::Item) -> bool + 'a,
    {
        self.indexed_iter()
            .filter(move |(_, val)| predicate(val))
            .map(|(idx, _)| idx)
    }

    fn fold_field<F, U>(&self, init: U, f: F) -> U
    where
        F: Fn(U, &Self::Item) -> U,
    {
        self.iter().fold(init, f)
    }

    fn scan_field<F, U>(&self, init: U, f: F) -> Array3<U>
    where
        F: Fn(&U, &Self::Item) -> U,
        U: Clone,
    {
        let shape = self.dim();
        let mut result = Array3::from_elem(shape, init.clone());
        let mut accumulator = init;

        for ((i, j, k), val) in self.indexed_iter() {
            accumulator = f(&accumulator, val);
            result[[i, j, k]] = accumulator.clone();
        }

        result
    }

    fn par_map_field<F, U>(&self, f: F) -> Array3<U>
    where
        F: Fn(&Self::Item) -> U + Sync + Send,
        U: Send + Sync,
        Self::Item: Sync,
    {
        let shape = self.dim();
        let flat_results: Vec<U> = self.iter().par_bridge().map(|x| f(x)).collect();

        Array3::from_shape_vec(shape, flat_results).expect("Shape mismatch in parallel map")
    }

    fn find_element<F>(&self, predicate: F) -> Option<((usize, usize, usize), &Self::Item)>
    where
        F: Fn(&Self::Item) -> bool,
    {
        self.indexed_iter().find(|(_, val)| predicate(val))
    }

    fn count_matching<F>(&self, predicate: F) -> usize
    where
        F: Fn(&Self::Item) -> bool,
    {
        self.iter().filter(|&val| predicate(val)).count()
    }

    fn any<F>(&self, predicate: F) -> bool
    where
        F: Fn(&Self::Item) -> bool,
    {
        self.iter().any(predicate)
    }

    fn all<F>(&self, predicate: F) -> bool
    where
        F: Fn(&Self::Item) -> bool,
    {
        self.iter().all(predicate)
    }
}

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

    // Pre-compute non-zero kernel elements
    let non_zero_kernel: Vec<((usize, usize, usize), K)> = kernel
        .indexed_iter()
        .filter(|(_, kval)| **kval != K::default())
        .map(|(idx, kval)| (idx, kval.clone()))
        .collect();

    // Create a flat vector of indices for parallel processing
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

/// Reduction operations over fields
pub trait FieldReduction<T> {
    /// Compute the sum of all elements
    fn sum(&self) -> T
    where
        T: std::iter::Sum;

    /// Compute the product of all elements
    fn product(&self) -> T
    where
        T: std::iter::Product;

    /// Find the minimum element
    fn min(&self) -> Option<&T>
    where
        T: Ord;

    /// Find the maximum element
    fn max(&self) -> Option<&T>
    where
        T: Ord;

    /// Compute the mean (average) of all elements
    fn mean(&self) -> T
    where
        T: std::iter::Sum + std::ops::Div<usize, Output = T> + Clone;
}

impl<T: Clone> FieldReduction<T> for Array3<T> {
    fn sum(&self) -> T
    where
        T: std::iter::Sum,
    {
        self.iter().cloned().sum()
    }

    fn product(&self) -> T
    where
        T: std::iter::Product,
    {
        self.iter().cloned().product()
    }

    fn min(&self) -> Option<&T>
    where
        T: Ord,
    {
        self.iter().min()
    }

    fn max(&self) -> Option<&T>
    where
        T: Ord,
    {
        self.iter().max()
    }

    fn mean(&self) -> T
    where
        T: std::iter::Sum + std::ops::Div<usize, Output = T> + Clone,
    {
        let sum: T = self.iter().cloned().sum();
        sum / self.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_filter_indices_lazy() {
        let field = Array3::from_shape_fn((3, 3, 3), |(i, j, k)| i + j + k);
        let indices: Vec<_> = field.filter_indices(|&x| x > 5).collect();

        assert!(!indices.is_empty());
        assert!(indices.contains(&(2, 2, 2)));
    }

    #[test]
    fn test_sparse_kernel() {
        let field = Array3::ones((5, 5, 5));
        let mut kernel = Array3::zeros((3, 3, 3));
        kernel[[1, 1, 1]] = 2.0; // Only center element is non-zero

        let result = apply_kernel(&field, &kernel, |f: &f64, k: &f64| f * k);
        assert_abs_diff_eq!(result[[2, 2, 2]], 2.0);
    }

    #[test]
    fn test_parallel_map() {
        let field = Array3::from_shape_fn((10, 10, 10), |(i, j, k)| (i + j + k) as f64);
        let result = field.par_map_field(|&x| x * 2.0);

        assert_abs_diff_eq!(result[[5, 5, 5]], (5.0 + 5.0 + 5.0) * 2.0);
    }

    #[test]
    fn test_field_reduction() {
        let field = Array3::from_shape_fn((2, 2, 2), |_| 3.0);

        assert_abs_diff_eq!(field.sum(), 24.0);
        assert_abs_diff_eq!(field.mean().unwrap(), 3.0);
        // Use fold to find max for f64 since it doesn't implement Ord
        let max_val = field.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        assert_abs_diff_eq!(max_val, 3.0);
    }

    #[test]
    fn test_windowed_operation() {
        let field = Array3::ones((5, 5, 5));
        let result = windowed_operation(&field, (3, 3, 3), |window| window.iter().sum::<f64>());

        // Center point should have full 3x3x3 = 27 neighbors
        assert_abs_diff_eq!(result[[2, 2, 2]], 27.0);
    }
}
