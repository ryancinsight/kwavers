//! In-place array operations for memory-efficient solver computations.

use moirai_parallel::{enumerate_mut_with, for_each_mut_with, Adaptive};
use leto::Array3;

/// Add two arrays in-place: `a += b`.
#[inline]
pub fn add_inplace(a: &mut Array3<f64>, b: &Array3<f64>) {
    assert_eq!(a.shape(), b.shape(), "invariant: add_inplace shape mismatch");
    match (a.as_slice_mut(), b.as_slice()) {
        (Some(a_slice), Some(b_slice)) => {
            enumerate_mut_with::<Adaptive, _, _>(a_slice, |idx, a| {
                *a += b_slice[idx];
            });
        }
        _ => a.iter_mut().zip(b.iter()).for_each(|(a, &b)| *a += b),
    }
}

/// Subtract two arrays in-place: `a -= b`.
#[inline]
pub fn sub_inplace(a: &mut Array3<f64>, b: &Array3<f64>) {
    assert_eq!(a.shape(), b.shape(), "invariant: sub_inplace shape mismatch");
    match (a.as_slice_mut(), b.as_slice()) {
        (Some(a_slice), Some(b_slice)) => {
            enumerate_mut_with::<Adaptive, _, _>(a_slice, |idx, a| {
                *a -= b_slice[idx];
            });
        }
        _ => a.iter_mut().zip(b.iter()).for_each(|(a, &b)| *a -= b),
    }
}

/// Multiply array by scalar in-place: `a *= scalar`.
#[inline]
pub fn scale_inplace(a: &mut Array3<f64>, scalar: f64) {
    if let Some(slice) = a.as_slice_mut() {
        for_each_mut_with::<Adaptive, _, _>(slice, |x| *x *= scalar);
    } else {
        for x in a.iter_mut() {
            *x *= scalar;
        }
    }
}

/// Compute `a = a * b + c` in-place (fused multiply-add).
#[inline]
pub fn fma_inplace(a: &mut Array3<f64>, b: &Array3<f64>, c: &Array3<f64>) {
    assert_eq!(a.shape(), b.shape(), "invariant: fma_inplace b shape mismatch");
    assert_eq!(a.shape(), c.shape(), "invariant: fma_inplace c shape mismatch");
    match (a.as_slice_mut(), b.as_slice(), c.as_slice()) {
        (Some(a_slice), Some(b_slice), Some(c_slice)) => {
            enumerate_mut_with::<Adaptive, _, _>(a_slice, |idx, a| {
                *a = (*a).mul_add(b_slice[idx], c_slice[idx]);
            });
        }
        _ => a
            .iter_mut()
            .zip(b.iter())
            .zip(c.iter())
            .for_each(|((a, &b), &c)| *a = (*a).mul_add(b, c)),
    }
}

/// Apply `f` to each element in-place.
#[inline]
pub fn apply_inplace<F>(a: &mut Array3<f64>, f: F)
where
    F: Fn(f64) -> f64 + Send + Sync,
{
    if let Some(slice) = a.as_slice_mut() {
        for_each_mut_with::<Adaptive, _, _>(slice, |x| *x = f(*x));
    } else {
        for x in a.iter_mut() {
            *x = f(*x);
        }
    }
}
