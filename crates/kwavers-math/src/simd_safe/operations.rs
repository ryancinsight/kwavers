//! Hermes-backed portable SIMD operations for dense `Array3<f64>` fields.
//!
//! All ISA dispatch is delegated to `hermes_simd`; this module contains only
//! the Array3 ↔ contiguous-slice bridge and the element-wise scalar fallback
//! for non-contiguous views.

use hermes_simd::{elementwise_add, elementwise_mul, elementwise_sub, scale};
use leto::Array3;

/// SIMD lane width for f64, selected at compile time per target ISA.
///
/// - x86_64 AVX2: 256 bits / 64 bits = 4 lanes
/// - AArch64 NEON: 128 bits / 64 bits = 2 lanes
/// - Scalar fallback: 1 lane
#[cfg(target_arch = "x86_64")]
pub const SIMD_WIDTH: usize = 4;

#[cfg(target_arch = "aarch64")]
pub const SIMD_WIDTH: usize = 2;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub const SIMD_WIDTH: usize = 1;

/// Portable SIMD operations backed by `hermes_simd`.
///
/// When the underlying `Array3` is contiguous in memory the work is dispatched
/// through the `hermes_simd` kernel (which handles runtime ISA selection
/// internally).  Non-contiguous views fall back to a safe scalar loop so that
/// correctness is preserved at all layout variants.
#[derive(Debug)]
pub struct SimdOps;

impl SimdOps {
    /// Add two fields element-wise.
    ///
    /// # Panics
    ///
    /// Panics if the contiguous inputs have different element counts. Equal
    /// shapes are required by the element-wise operation; the Hermes kernel
    /// reports a mismatch and this invariant failure is propagated as a panic.
    #[inline]
    #[must_use]
    pub fn add_fields(a: &Array3<f64>, b: &Array3<f64>) -> Array3<f64> {
        let shape = a.shape();
        let mut result = Array3::zeros(shape);
        if let (Some(a_s), Some(b_s), Some(r_s)) = (
            a.as_slice_memory_order(),
            b.as_slice_memory_order(),
            result.as_slice_memory_order_mut(),
        ) {
            elementwise_add(a_s, b_s, r_s)
                .expect("invariant: equal Array3 shapes produce equal slice lengths");
        } else {
            for ((r, &av), &bv) in result.iter_mut().zip(a.iter()).zip(b.iter()) {
                *r = av + bv;
            }
        }
        result
    }

    /// Scale field by scalar.
    #[inline]
    #[must_use]
    pub fn scale_field(field: &Array3<f64>, scalar: f64) -> Array3<f64> {
        let mut result = field.clone();
        if let Some(s) = result.as_slice_memory_order_mut() {
            scale(s, scalar);
        } else {
            for v in result.iter_mut() {
                *v *= scalar;
            }
        }
        result
    }

    /// Compute the L2 (Euclidean) norm of a field.
    ///
    /// # Panics
    ///
    /// Panics if Hermes rejects the dot product because its two operands have
    /// different lengths. Both operands are the same contiguous slice, so this
    /// indicates a violated internal invariant.
    #[inline]
    #[must_use]
    pub fn norm(field: &Array3<f64>) -> f64 {
        if let Some(s) = field.as_slice_memory_order() {
            hermes_simd::dot(s, s)
                .expect("invariant: same-slice dot never mismatches length")
                .sqrt()
        } else {
            field.iter().map(|&v| v * v).sum::<f64>().sqrt()
        }
    }

    /// Multiply two fields element-wise.
    ///
    /// # Panics
    ///
    /// Panics if the contiguous inputs have different element counts. Equal
    /// shapes are required by the element-wise operation; the Hermes kernel
    /// reports a mismatch and this invariant failure is propagated as a panic.
    #[inline]
    #[must_use]
    pub fn multiply_fields(a: &Array3<f64>, b: &Array3<f64>) -> Array3<f64> {
        let shape = a.shape();
        let mut result = Array3::zeros(shape);
        if let (Some(a_s), Some(b_s), Some(r_s)) = (
            a.as_slice_memory_order(),
            b.as_slice_memory_order(),
            result.as_slice_memory_order_mut(),
        ) {
            elementwise_mul(a_s, b_s, r_s)
                .expect("invariant: equal Array3 shapes produce equal slice lengths");
        } else {
            for ((r, &av), &bv) in result.iter_mut().zip(a.iter()).zip(b.iter()) {
                *r = av * bv;
            }
        }
        result
    }

    /// Subtract `b` from `a` element-wise.
    ///
    /// # Panics
    ///
    /// Panics if the contiguous inputs have different element counts. Equal
    /// shapes are required by the element-wise operation; the Hermes kernel
    /// reports a mismatch and this invariant failure is propagated as a panic.
    #[inline]
    #[must_use]
    pub fn subtract_fields(a: &Array3<f64>, b: &Array3<f64>) -> Array3<f64> {
        let shape = a.shape();
        let mut result = Array3::zeros(shape);
        if let (Some(a_s), Some(b_s), Some(r_s)) = (
            a.as_slice_memory_order(),
            b.as_slice_memory_order(),
            result.as_slice_memory_order_mut(),
        ) {
            elementwise_sub(a_s, b_s, r_s)
                .expect("invariant: equal Array3 shapes produce equal slice lengths");
        } else {
            for ((r, &av), &bv) in result.iter_mut().zip(a.iter()).zip(b.iter()) {
                *r = av - bv;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eunomia::assert_relative_eq;

    #[test]
    fn test_add_fields_correctness() {
        let a = Array3::from_elem((10, 10, 10), 2.0);
        let b = Array3::from_elem((10, 10, 10), 3.0);
        let result = SimdOps::add_fields(&a, &b);

        for &val in result.iter() {
            assert_relative_eq!(val, 5.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_scale_field_correctness() {
        let field = Array3::from_elem((10, 10, 10), 2.0);
        let result = SimdOps::scale_field(&field, 3.0);

        for &val in result.iter() {
            assert_relative_eq!(val, 6.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_norm_correctness() {
        let field = Array3::from_elem((10, 10, 10), 1.0);
        let norm = SimdOps::norm(&field);
        assert_relative_eq!(norm, (1000.0_f64).sqrt(), epsilon = 1e-10);
    }
}
