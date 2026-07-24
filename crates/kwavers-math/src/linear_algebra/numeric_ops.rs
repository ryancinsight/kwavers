//! Generic numeric operations - SSOT: leto_ops element-wise ops and reductions.
//!
//! NumericOps is a thin trait over element-wise vector operations on Array1.

use eunomia::RealField;
use leto::Array1;

/// Generic numeric operations backed by Array1 direct operations.
pub trait NumericOps<T>: Copy + PartialOrd
where
    T: RealField,
{
    /// Dot product of two rank-1 arrays.
    fn dot(a: &Array1<T>, b: &Array1<T>) -> T;

    /// Element-wise addition.
    fn add_elementwise(a: &Array1<T>, b: &Array1<T>) -> Array1<T>;

    /// Element-wise multiplication.
    fn mul_elementwise(a: &Array1<T>, b: &Array1<T>) -> Array1<T>;

    /// Safe division with zero-guard.
    fn safe_divide(a: &Array1<T>, b: &Array1<T>, default: T) -> Array1<T>;
}

impl<T: RealField> NumericOps<T> for T {
    fn dot(a: &Array1<T>, b: &Array1<T>) -> T {
        assert_eq!(a.shape(), b.shape(), "dot: length mismatch");
        let n = a.shape()[0];
        (0..n).fold(T::ZERO, |acc, i| acc + a[i] * b[i])
    }

    fn add_elementwise(a: &Array1<T>, b: &Array1<T>) -> Array1<T> {
        assert_eq!(a.shape(), b.shape(), "add_elementwise: length mismatch");
        let n = a.shape()[0];
        Array1::from_shape_vec([n], (0..n).map(|i| a[i] + b[i]).collect())
            .expect("shape matches")
    }

    fn mul_elementwise(a: &Array1<T>, b: &Array1<T>) -> Array1<T> {
        assert_eq!(a.shape(), b.shape(), "mul_elementwise: length mismatch");
        let n = a.shape()[0];
        Array1::from_shape_vec([n], (0..n).map(|i| a[i] * b[i]).collect())
            .expect("shape matches")
    }

    fn safe_divide(a: &Array1<T>, b: &Array1<T>, default: T) -> Array1<T> {
        assert_eq!(a.shape(), b.shape(), "safe_divide: length mismatch");
        let n = a.shape()[0];
        Array1::from_shape_vec(
            [n],
            (0..n).map(|i| if b[i].abs() < T::EPSILON { default } else { a[i] / b[i] }).collect(),
        ).expect("shape matches")
    }
}