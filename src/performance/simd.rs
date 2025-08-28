//! SIMD optimizations for field operations
//!
//! Provides vectorized implementations of common field operations
//! for significant performance improvements.

// SIMD requires unsafe for performance - all unsafe blocks are justified
#![allow(unsafe_code)]

use ndarray::{Array3, Zip};
use std::arch::x86_64::*;

/// SIMD-optimized field operations
#[derive(Debug)]
pub struct SimdOps;

impl SimdOps {
    /// Vectorized field addition using AVX2
    ///
    /// # Safety
    /// Requires AVX2 CPU support. Falls back to scalar on older CPUs.
    #[cfg(target_arch = "x86_64")]
    pub fn add_fields(a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
        if is_x86_feature_detected!("avx2") {
            unsafe { Self::add_fields_avx2(a, b, out) }
        } else {
            Self::add_fields_scalar(a, b, out)
        }
    }

    /// Scalar fallback for field addition
    #[cfg(not(target_arch = "x86_64"))]
    pub fn add_fields(a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
        Self::add_fields_scalar(a, b, out)
    }

    /// AVX2 implementation of field addition
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn add_fields_avx2(a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
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

    /// Scalar implementation of field addition
    fn add_fields_scalar(a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
        Zip::from(out)
            .and(a)
            .and(b)
            .for_each(|o, &a, &b| *o = a + b);
    }

    /// Vectorized field multiplication by scalar
    #[cfg(target_arch = "x86_64")]
    pub fn scale_field(field: &Array3<f64>, scalar: f64, out: &mut Array3<f64>) {
        if is_x86_feature_detected!("avx2") {
            unsafe { Self::scale_field_avx2(field, scalar, out) }
        } else {
            Self::scale_field_scalar(field, scalar, out)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn scale_field(field: &Array3<f64>, scalar: f64, out: &mut Array3<f64>) {
        Self::scale_field_scalar(field, scalar, out)
    }

    /// AVX2 implementation of field scaling
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn scale_field_avx2(field: &Array3<f64>, scalar: f64, out: &mut Array3<f64>) {
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

    fn scale_field_scalar(field: &Array3<f64>, scalar: f64, out: &mut Array3<f64>) {
        Zip::from(out).and(field).for_each(|o, &f| *o = f * scalar);
    }

    /// Compute L2 norm of field using SIMD
    #[cfg(target_arch = "x86_64")]
    pub fn field_norm(field: &Array3<f64>) -> f64 {
        if is_x86_feature_detected!("avx2") {
            unsafe { Self::field_norm_avx2(field) }
        } else {
            Self::field_norm_scalar(field)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn field_norm(field: &Array3<f64>) -> f64 {
        Self::field_norm_scalar(field)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn field_norm_avx2(field: &Array3<f64>) -> f64 {
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
