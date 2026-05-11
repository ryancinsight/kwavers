//! In-place array operations for memory-efficient solver computations.

use ndarray::{Array3, Zip};

/// Add two arrays in-place: `a += b`.
#[inline]
pub fn add_inplace(a: &mut Array3<f64>, b: &Array3<f64>) {
    Zip::from(a).and(b).par_for_each(|a, &b| *a += b);
}

/// Subtract two arrays in-place: `a -= b`.
#[inline]
pub fn sub_inplace(a: &mut Array3<f64>, b: &Array3<f64>) {
    Zip::from(a).and(b).par_for_each(|a, &b| *a -= b);
}

/// Multiply array by scalar in-place: `a *= scalar`.
#[inline]
pub fn scale_inplace(a: &mut Array3<f64>, scalar: f64) {
    a.par_mapv_inplace(|x| x * scalar);
}

/// Compute `a = a * b + c` in-place (fused multiply-add).
#[inline]
pub fn fma_inplace(a: &mut Array3<f64>, b: &Array3<f64>, c: &Array3<f64>) {
    Zip::from(a)
        .and(b)
        .and(c)
        .for_each(|a, &b, &c| *a = (*a).mul_add(b, c));
}

/// Apply `f` to each element in-place.
#[inline]
pub fn apply_inplace<F>(a: &mut Array3<f64>, f: F)
where
    F: Fn(f64) -> f64 + Send + Sync,
{
    a.par_mapv_inplace(f);
}
