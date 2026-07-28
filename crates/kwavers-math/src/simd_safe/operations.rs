//! Portable SIMD field operations backed by `hermes-simd` runtime dispatch.
//!
//! Each method extracts a contiguous slice from the `Array3` and forwards to
//! the corresponding `hermes_simd` free function, which selects AVX-512,
//! AVX2, NEON, or scalar at runtime.  All unsafe intrinsics are encapsulated
//! inside `hermes_simd_intrinsics`; this module stays `#[forbid(unsafe_code)]`.

use hermes_simd::{axpy, dot, elementwise_add, elementwise_mul, elementwise_sub, scale};
use leto::Array3;

/// Informational: expected maximum SIMD lane count for `f64` on this target.
#[cfg(target_arch = "x86_64")]
pub const SIMD_WIDTH: usize = 4; // AVX2 256-bit / 64-bit = 4 lanes

#[cfg(target_arch = "aarch64")]
pub const SIMD_WIDTH: usize = 2; // NEON 128-bit / 64-bit = 2 lanes

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub const SIMD_WIDTH: usize = 1;

/// Portable SIMD field operations.
///
/// All methods delegate to `hermes_simd` for AVX-512 / AVX2 / NEON / scalar
/// runtime dispatch.
#[derive(Debug)]
pub struct SimdOps;

impl SimdOps {
    /// Add two 3-D fields element-wise.
    #[inline]
    #[must_use]
    pub fn add_fields(a: &Array3<f64>, b: &Array3<f64>) -> Array3<f64> {
        let shape = a.shape();
        let mut result = Array3::zeros(shape);
        if let (Some(a_s), Some(b_s), Some(out)) =
            (a.as_slice(), b.as_slice(), result.as_slice_mut())
        {
            elementwise_add(a_s, b_s, out).expect("add_fields: shape mismatch");
        }
        result
    }

    /// Scale a 3-D field by a scalar, returning a new owned array.
    #[inline]
    #[must_use]
    pub fn scale_field(field: &Array3<f64>, scalar: f64) -> Array3<f64> {
        let shape = field.shape();
        let mut result = Array3::zeros(shape);
        if let (Some(src), Some(out)) = (field.as_slice(), result.as_slice_mut()) {
            out.copy_from_slice(src);
            scale(out, scalar);
        }
        result
    }

    /// L2 norm of a 3-D field: `√(Σ xᵢ²)`.
    #[inline]
    #[must_use]
    pub fn norm(field: &Array3<f64>) -> f64 {
        field
            .as_slice()
            .and_then(|s| dot(s, s).ok())
            .map(f64::sqrt)
            .unwrap_or(0.0)
    }

    /// Multiply two 3-D fields element-wise.
    #[inline]
    #[must_use]
    pub fn multiply_fields(a: &Array3<f64>, b: &Array3<f64>) -> Array3<f64> {
        let shape = a.shape();
        let mut result = Array3::zeros(shape);
        if let (Some(a_s), Some(b_s), Some(out)) =
            (a.as_slice(), b.as_slice(), result.as_slice_mut())
        {
            elementwise_mul(a_s, b_s, out).expect("multiply_fields: shape mismatch");
        }
        result
    }

    /// Subtract two 3-D fields element-wise (`a − b`).
    #[inline]
    #[must_use]
    pub fn subtract_fields(a: &Array3<f64>, b: &Array3<f64>) -> Array3<f64> {
        let shape = a.shape();
        let mut result = Array3::zeros(shape);
        if let (Some(a_s), Some(b_s), Some(out)) =
            (a.as_slice(), b.as_slice(), result.as_slice_mut())
        {
            elementwise_sub(a_s, b_s, out).expect("subtract_fields: shape mismatch");
        }
        result
    }

    /// Compute `y += alpha * x` on 3-D fields in-place.
    ///
    /// Convenience wrapper around `hermes_simd::axpy` for field operations.
    #[inline]
    pub fn axpy_fields(alpha: f64, x: &Array3<f64>, y: &mut Array3<f64>) {
        if let (Some(x_s), Some(y_s)) = (x.as_slice(), y.as_slice_mut()) {
            axpy(alpha, x_s, y_s).expect("axpy_fields: shape mismatch");
        }
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
