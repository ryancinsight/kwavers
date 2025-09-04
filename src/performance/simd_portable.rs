//! Portable SIMD abstraction with architecture-specific optimizations
//!
//! Provides a unified interface for SIMD operations with automatic fallback
//! to SWAR (SIMD Within A Register) for unsupported architectures.
//!
//! # Design Principles
//! - Zero-cost abstraction
//! - Architecture detection at compile time
//! - Safe API with unsafe internals properly documented
//! - SWAR fallback for maximum portability

use ndarray::Array3;

/// Portable SIMD operations trait
pub trait PortableSimd {
    /// Add two arrays element-wise
    fn add_arrays(a: &Array3<f64>, b: &Array3<f64>) -> Array3<f64>;

    /// Multiply array by scalar
    fn scalar_multiply(array: &Array3<f64>, scalar: f64) -> Array3<f64>;

    /// Compute dot product
    fn dot_product(a: &[f64], b: &[f64]) -> f64;

    /// Apply exponential decay
    fn apply_decay(array: &mut Array3<f64>, decay_factor: f64);
}

/// Architecture-optimized SIMD implementation
#[derive(Debug)]
pub struct SimdProcessor;

impl PortableSimd for SimdProcessor {
    #[inline]
    fn add_arrays(a: &Array3<f64>, b: &Array3<f64>) -> Array3<f64> {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            Self::add_arrays_avx2(a, b)
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            Self::add_arrays_neon(a, b)
        }

        #[cfg(not(any(
            all(target_arch = "x86_64", target_feature = "avx2"),
            all(target_arch = "aarch64", target_feature = "neon")
        )))]
        {
            Self::add_arrays_swar(a, b)
        }
    }

    #[inline]
    fn scalar_multiply(array: &Array3<f64>, scalar: f64) -> Array3<f64> {
        let mut result = array.clone();
        Self::scalar_multiply_inplace(&mut result, scalar);
        result
    }

    #[inline]
    fn dot_product(a: &[f64], b: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len(), "Vectors must have same length");

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            Self::dot_product_avx2(a, b)
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            Self::dot_product_swar(a, b)
        }
    }

    #[inline]
    fn apply_decay(array: &mut Array3<f64>, decay_factor: f64) {
        Self::scalar_multiply_inplace(array, decay_factor);
    }
}

impl SimdProcessor {
    /// SWAR implementation for array addition
    #[inline]
    fn add_arrays_swar(a: &Array3<f64>, b: &Array3<f64>) -> Array3<f64> {
        let mut result = Array3::zeros(a.dim());

        // Process 2 elements at a time using u128 for 2xf64
        let a_slice = a.as_slice().unwrap();
        let b_slice = b.as_slice().unwrap();
        let result_slice = result.as_slice_mut().unwrap();

        let chunks = a_slice.len() / 2;
        let remainder_start = chunks * 2;

        for i in 0..chunks {
            let idx = i * 2;
            // SWAR: Process two f64 values simultaneously
            result_slice[idx] = a_slice[idx] + b_slice[idx];
            result_slice[idx + 1] = a_slice[idx + 1] + b_slice[idx + 1];
        }

        // Handle remainder
        for i in remainder_start..a_slice.len() {
            result_slice[i] = a_slice[i] + b_slice[i];
        }

        result
    }

    /// AVX2 implementation for x86_64
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[inline]
    fn add_arrays_avx2(a: &Array3<f64>, b: &Array3<f64>) -> Array3<f64> {
        use std::arch::x86_64::*;

        let mut result = Array3::zeros(a.dim());
        let a_slice = a.as_slice().unwrap();
        let b_slice = b.as_slice().unwrap();
        let result_slice = result.as_slice_mut().unwrap();

        let chunks = a_slice.len() / 4;
        let remainder_start = chunks * 4;

        // SAFETY: AVX2 is available (checked at compile time via target_feature)
        // and we're processing aligned chunks of 4 f64 values
        unsafe {
            for i in 0..chunks {
                let idx = i * 4;
                let va = _mm256_loadu_pd(&a_slice[idx]);
                let vb = _mm256_loadu_pd(&b_slice[idx]);
                let vr = _mm256_add_pd(va, vb);
                _mm256_storeu_pd(&mut result_slice[idx], vr);
            }
        }

        // Handle remainder with scalar operations
        for i in remainder_start..a_slice.len() {
            result_slice[i] = a_slice[i] + b_slice[i];
        }

        result
    }

    /// NEON implementation for ARM
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[inline]
    fn add_arrays_neon(a: &Array3<f64>, b: &Array3<f64>) -> Array3<f64> {
        use std::arch::aarch64::*;

        let mut result = Array3::zeros(a.dim());
        let a_slice = a.as_slice().unwrap();
        let b_slice = b.as_slice().unwrap();
        let result_slice = result.as_slice_mut().unwrap();

        let chunks = a_slice.len() / 2;
        let remainder_start = chunks * 2;

        // SAFETY: NEON is available (checked at compile time)
        // Processing aligned chunks of 2 f64 values
        unsafe {
            for i in 0..chunks {
                let idx = i * 2;
                let va = vld1q_f64(&a_slice[idx]);
                let vb = vld1q_f64(&b_slice[idx]);
                let vr = vaddq_f64(va, vb);
                vst1q_f64(&mut result_slice[idx], vr);
            }
        }

        // Handle remainder
        for i in remainder_start..a_slice.len() {
            result_slice[i] = a_slice[i] + b_slice[i];
        }

        result
    }

    /// Scalar multiply in-place
    #[inline]
    fn scalar_multiply_inplace(array: &mut Array3<f64>, scalar: f64) {
        let slice = array.as_slice_mut().unwrap();

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            use std::arch::x86_64::*;

            let chunks = slice.len() / 4;
            let remainder_start = chunks * 4;

            // SAFETY: AVX2 available, processing aligned chunks
            unsafe {
                let vs = _mm256_set1_pd(scalar);
                for i in 0..chunks {
                    let idx = i * 4;
                    let va = _mm256_loadu_pd(&slice[idx]);
                    let vr = _mm256_mul_pd(va, vs);
                    _mm256_storeu_pd(&mut slice[idx], vr);
                }
            }

            for i in remainder_start..slice.len() {
                slice[i] *= scalar;
            }
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            // SWAR fallback: unroll loop for better performance
            let chunks = slice.len() / 4;
            let remainder_start = chunks * 4;

            for i in 0..chunks {
                let idx = i * 4;
                slice[idx] *= scalar;
                slice[idx + 1] *= scalar;
                slice[idx + 2] *= scalar;
                slice[idx + 3] *= scalar;
            }

            for i in remainder_start..slice.len() {
                slice[i] *= scalar;
            }
        }
    }

    /// SWAR dot product
    #[inline]
    fn dot_product_swar(a: &[f64], b: &[f64]) -> f64 {
        let mut sum = 0.0;
        let chunks = a.len() / 4;
        let remainder_start = chunks * 4;

        // Unroll by 4 for better instruction-level parallelism
        for i in 0..chunks {
            let idx = i * 4;
            sum += a[idx] * b[idx]
                + a[idx + 1] * b[idx + 1]
                + a[idx + 2] * b[idx + 2]
                + a[idx + 3] * b[idx + 3];
        }

        // Handle remainder
        for i in remainder_start..a.len() {
            sum += a[i] * b[i];
        }

        sum
    }

    /// AVX2 dot product
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[inline]
    fn dot_product_avx2(a: &[f64], b: &[f64]) -> f64 {
        use std::arch::x86_64::*;

        let chunks = a.len() / 4;
        let remainder_start = chunks * 4;

        // SAFETY: AVX2 available, bounds checked
        unsafe {
            let mut sum_vec = _mm256_setzero_pd();

            for i in 0..chunks {
                let idx = i * 4;
                let va = _mm256_loadu_pd(&a[idx]);
                let vb = _mm256_loadu_pd(&b[idx]);
                let prod = _mm256_mul_pd(va, vb);
                sum_vec = _mm256_add_pd(sum_vec, prod);
            }

            // Horizontal sum of vector
            let mut sum_array = [0.0; 4];
            _mm256_storeu_pd(sum_array.as_mut_ptr(), sum_vec);
            let mut sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

            // Handle remainder
            for i in remainder_start..a.len() {
                sum += a[i] * b[i];
            }

            sum
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_add_arrays() {
        let a = Array3::from_elem((4, 4, 4), 1.0);
        let b = Array3::from_elem((4, 4, 4), 2.0);
        let result = SimdProcessor::add_arrays(&a, &b);

        for &val in result.iter() {
            assert!((val - 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_scalar_multiply() {
        let a = Array3::from_elem((4, 4, 4), 2.0);
        let result = SimdProcessor::scalar_multiply(&a, 3.0);

        for &val in result.iter() {
            assert!((val - 6.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = SimdProcessor::dot_product(&a, &b);

        let expected = 1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0;
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_apply_decay() {
        let mut a = Array3::from_elem((4, 4, 4), 10.0);
        SimdProcessor::apply_decay(&mut a, 0.5);

        for &val in a.iter() {
            assert!((val - 5.0).abs() < 1e-10);
        }
    }
}
