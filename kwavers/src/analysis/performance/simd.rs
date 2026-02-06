//! SIMD optimizations for field operations
//!
//! Provides vectorized implementations of common field operations
//! for significant performance improvements.

// SIMD requires unsafe for performance - all unsafe blocks are justified
#![allow(unsafe_code)]

use ndarray::{Array3, Zip};
use std::arch::x86_64::{
    _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_setzero_pd,
    _mm256_storeu_pd,
};

/// SIMD-optimized field operations
#[derive(Debug)]
pub struct SimdOps;

impl SimdOps {
    /// Vectorized field addition using AVX2
    ///
    /// # Safety
    /// Portable SIMD-aware field addition using iterator combinators
    pub fn add_fields(a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
        // Use safe iteration for non-contiguous arrays
        if let (Some(out_slice), Some(a_slice), Some(b_slice)) =
            (out.as_slice_mut(), a.as_slice(), b.as_slice())
        {
            // Use iterator combinators for auto-vectorization
            out_slice
                .iter_mut()
                .zip(a_slice)
                .zip(b_slice)
                .for_each(|((o, &a_val), &b_val)| {
                    *o = a_val + b_val;
                });
        } else {
            // Fallback for non-contiguous arrays
            out.iter_mut()
                .zip(a.iter())
                .zip(b.iter())
                .for_each(|((o, &a_val), &b_val)| {
                    *o = a_val + b_val;
                });
        }
    }

    /// Legacy AVX2 implementation - replaced with portable SIMD
    #[allow(dead_code)]
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn add_fields_avx2_legacy(a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
        // SAFETY: This function requires AVX2 to be available, enforced by #[target_feature].
        // Array slices are guaranteed to be properly aligned and non-overlapping.
        // The function operates on contiguous memory regions with valid f64 values.
        unsafe {
            let a_slice = a.as_slice().unwrap();
            let b_slice = b.as_slice().unwrap();
            let out_slice = out.as_slice_mut().unwrap();

            let chunks = a_slice.len() / 4;
            let remainder = a_slice.len() % 4;

            // Process 4 doubles at a time with AVX2
            for i in 0..chunks {
                let idx = i * 4;
                let va = _mm256_loadu_pd(&a_slice[idx]);
                let vb = _mm256_loadu_pd(&b_slice[idx]);
                let result = _mm256_add_pd(va, vb);
                _mm256_storeu_pd(&mut out_slice[idx], result);
            }

            // Handle remainder
            let start = chunks * 4;
            for i in 0..remainder {
                out_slice[start + i] = a_slice[start + i] + b_slice[start + i];
            }
        }
    }

    /// Scalar implementation of field addition
    /// Scalar fallback implementation
    #[allow(dead_code)]
    fn add_fields_scalar(a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
        Zip::from(out)
            .and(a)
            .and(b)
            .for_each(|o, &a, &b| *o = a + b);
    }

    /// Vectorized field multiplication by scalar
    /// Portable SIMD-aware field scaling using iterator combinators
    pub fn scale_field(field: &Array3<f64>, scalar: f64, out: &mut Array3<f64>) {
        // Use iterator combinators for auto-vectorization
        out.as_slice_mut()
            .unwrap()
            .iter_mut()
            .zip(field.as_slice().unwrap())
            .for_each(|(o, &f_val)| {
                *o = f_val * scalar;
            });
    }

    /// AVX2 implementation of field scaling
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    /// AVX2 field scaling
    #[allow(dead_code)]
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn scale_field_avx2(field: &Array3<f64>, scalar: f64, out: &mut Array3<f64>) {
        // SAFETY: This function requires AVX2 to be available, enforced by #[target_feature].
        // Input and output arrays must have the same dimensions and be properly aligned.
        // The scalar broadcast operation is safe for all finite f64 values.
        unsafe {
            let field_slice = field.as_slice().unwrap();
            let out_slice = out.as_slice_mut().unwrap();

            let scalar_vec = _mm256_set1_pd(scalar);
            let chunks = field_slice.len() / 4;
            let remainder = field_slice.len() % 4;

            for i in 0..chunks {
                let idx = i * 4;
                let v = _mm256_loadu_pd(&field_slice[idx]);
                let result = _mm256_mul_pd(v, scalar_vec);
                _mm256_storeu_pd(&mut out_slice[idx], result);
            }

            let start = chunks * 4;
            for i in 0..remainder {
                out_slice[start + i] = field_slice[start + i] * scalar;
            }
        }
    }

    /// Scalar field scaling fallback
    #[allow(dead_code)]
    fn scale_field_scalar(field: &Array3<f64>, scalar: f64, out: &mut Array3<f64>) {
        Zip::from(out).and(field).for_each(|o, &f| *o = f * scalar);
    }

    /// Compute L2 norm of field using SIMD
    /// Portable SIMD-aware field norm using iterator combinators
    #[must_use]
    pub fn field_norm(field: &Array3<f64>) -> f64 {
        // Use iterator combinators for auto-vectorization
        field
            .as_slice()
            .unwrap()
            .iter()
            .map(|&x| x * x)
            .sum::<f64>()
            .sqrt()
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    /// AVX2 field norm computation
    #[allow(dead_code)]
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn field_norm_avx2(field: &Array3<f64>) -> f64 {
        // SAFETY: This function requires AVX2 to be available, enforced by #[target_feature].
        // Array slice is guaranteed to be contiguous and properly aligned.
        // Bounds checking: chunks*4 <= slice.len() by construction, remainder handled separately.
        // Memory safety: slice data valid throughout function execution via borrow checker.
        unsafe {
            let slice = field.as_slice().unwrap();
            let chunks = slice.len() / 4;
            let remainder = slice.len() % 4;

            let mut sum = _mm256_setzero_pd();

            for i in 0..chunks {
                let idx = i * 4;
                let v = _mm256_loadu_pd(&slice[idx]);
                let squared = _mm256_mul_pd(v, v);
                sum = _mm256_add_pd(sum, squared);
            }

            // Sum the 4 doubles in the AVX register
            let mut result = [0.0; 4];
            _mm256_storeu_pd(&mut result[0], sum);
            let mut total = result[0] + result[1] + result[2] + result[3];

            // Handle remainder
            let start = chunks * 4;
            for i in 0..remainder {
                total += slice[start + i] * slice[start + i];
            }

            total.sqrt()
        }
    }

    /// Scalar field norm fallback
    #[allow(dead_code)]
    fn field_norm_scalar(field: &Array3<f64>) -> f64 {
        field.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_simd_addition() {
        let a = Array3::from_elem((10, 10, 10), 1.0);
        let b = Array3::from_elem((10, 10, 10), 1.0) * 2.0;
        let mut out = Array3::zeros((10, 10, 10));

        SimdOps::add_fields(&a, &b, &mut out);

        // Should be 3.0 everywhere
        assert!((out[[5, 5, 5]] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_simd_scaling() {
        let field = Array3::from_elem((10, 10, 10), 1.0) * 2.0;
        let mut out = Array3::zeros((10, 10, 10));

        SimdOps::scale_field(&field, 3.0, &mut out);

        // Should be 6.0 everywhere
        assert!((out[[5, 5, 5]] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_simd_norm() {
        let mut field = Array3::zeros((10, 10, 10));
        field[[0, 0, 0]] = 3.0;
        field[[0, 0, 1]] = 4.0;

        let norm = SimdOps::field_norm(&field);

        // Should be 5.0 (3-4-5 triangle)
        assert!((norm - 5.0).abs() < 1e-10);
    }
}
