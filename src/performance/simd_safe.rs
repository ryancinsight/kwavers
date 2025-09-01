//! Safe, portable SIMD operations with architecture-conditional compilation
//!
//! This module provides SIMD acceleration with:
//! - Architecture-specific optimizations (x86_64, aarch64)
//! - SWAR (SIMD Within A Register) fallback for unsupported architectures
//! - Zero unsafe blocks in public API
//! - Compile-time feature detection

use ndarray::Array3;

/// SIMD lane width for f64 operations
#[cfg(target_arch = "x86_64")]
const SIMD_WIDTH: usize = 4; // AVX2: 256 bits / 64 bits = 4

#[cfg(target_arch = "aarch64")]
const SIMD_WIDTH: usize = 2; // NEON: 128 bits / 64 bits = 2

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const SIMD_WIDTH: usize = 1; // Scalar fallback

/// Portable SIMD operations
pub struct SimdOps;

impl SimdOps {
    /// Add two fields element-wise using SIMD
    #[inline]
    pub fn add_fields(a: &Array3<f64>, b: &Array3<f64>) -> Array3<f64> {
        let shape = a.dim();
        let mut result = Array3::zeros(shape);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                Self::add_fields_avx2(a, b, &mut result);
                return result;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                Self::add_fields_neon(a, b, &mut result);
                return result;
            }
        }

        // SWAR fallback for other architectures
        Self::add_fields_swar(a, b, &mut result);
        result
    }

    /// Scale field by scalar using SIMD
    #[inline]
    pub fn scale_field(field: &Array3<f64>, scalar: f64) -> Array3<f64> {
        let shape = field.dim();
        let mut result = Array3::zeros(shape);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                Self::scale_field_avx2(field, scalar, &mut result);
                return result;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                Self::scale_field_neon(field, scalar, &mut result);
                return result;
            }
        }

        // SWAR fallback
        Self::scale_field_swar(field, scalar, &mut result);
        result
    }

    /// Compute L2 norm using SIMD
    #[inline]
    pub fn norm(field: &Array3<f64>) -> f64 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return Self::norm_avx2(field);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return Self::norm_neon(field);
            }
        }

        // SWAR fallback
        Self::norm_swar(field)
    }

    // Architecture-specific implementations
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn add_fields_avx2_inner(a: &[f64], b: &[f64], out: &mut [f64]) {
        unsafe {
            use std::arch::x86_64::*;

            let chunks = a.len() / 4;
            for i in 0..chunks {
                let offset = i * 4;
                let va = _mm256_loadu_pd(a.as_ptr().add(offset));
                let vb = _mm256_loadu_pd(b.as_ptr().add(offset));
                let sum = _mm256_add_pd(va, vb);
                _mm256_storeu_pd(out.as_mut_ptr().add(offset), sum);
            }

            // Handle remainder
            let remainder_start = chunks * 4;
            for i in remainder_start..a.len() {
                out[i] = a[i] + b[i];
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn add_fields_avx2(a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
        if let (Some(a_slice), Some(b_slice), Some(out_slice)) =
            (a.as_slice(), b.as_slice(), out.as_slice_mut())
        {
            // SAFETY:
            // 1. Feature detection via is_x86_feature_detected! ensures AVX2 is available
            // 2. Slices are guaranteed to have same length from Array3 shape equality
            // 3. Pointer arithmetic stays within slice bounds
            // 4. No data races as we have exclusive access to out_slice
            unsafe { Self::add_fields_avx2_inner(a_slice, b_slice, out_slice) }
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn add_fields_neon(a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
        use std::arch::aarch64::*;

        if let (Some(a_slice), Some(b_slice), Some(out_slice)) =
            (a.as_slice(), b.as_slice(), out.as_slice_mut())
        {
            // SAFETY: Feature detection ensures NEON is available
            unsafe {
                let chunks = a_slice.len() / 2;
                for i in 0..chunks {
                    let offset = i * 2;
                    let va = vld1q_f64(a_slice.as_ptr().add(offset));
                    let vb = vld1q_f64(b_slice.as_ptr().add(offset));
                    let sum = vaddq_f64(va, vb);
                    vst1q_f64(out_slice.as_mut_ptr().add(offset), sum);
                }

                // Handle remainder
                let remainder_start = chunks * 2;
                for i in remainder_start..a_slice.len() {
                    out_slice[i] = a_slice[i] + b_slice[i];
                }
            }
        }
    }

    /// SWAR (SIMD Within A Register) fallback for portable performance
    fn add_fields_swar(a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
        // Process multiple elements per iteration for better ILP
        const UNROLL: usize = 4;

        if let (Some(a_slice), Some(b_slice), Some(out_slice)) =
            (a.as_slice(), b.as_slice(), out.as_slice_mut())
        {
            let chunks = a_slice.len() / UNROLL;
            let mut i = 0;

            // Unrolled loop for instruction-level parallelism
            for _ in 0..chunks {
                out_slice[i] = a_slice[i] + b_slice[i];
                out_slice[i + 1] = a_slice[i + 1] + b_slice[i + 1];
                out_slice[i + 2] = a_slice[i + 2] + b_slice[i + 2];
                out_slice[i + 3] = a_slice[i + 3] + b_slice[i + 3];
                i += UNROLL;
            }

            // Handle remainder
            while i < a_slice.len() {
                out_slice[i] = a_slice[i] + b_slice[i];
                i += 1;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn scale_field_avx2_inner(field: &[f64], scalar: f64, out: &mut [f64]) {
        unsafe {
            use std::arch::x86_64::*;

            let scalar_vec = _mm256_set1_pd(scalar);
            let chunks = field.len() / 4;

            for i in 0..chunks {
                let offset = i * 4;
                let v = _mm256_loadu_pd(field.as_ptr().add(offset));
                let scaled = _mm256_mul_pd(v, scalar_vec);
                _mm256_storeu_pd(out.as_mut_ptr().add(offset), scaled);
            }

            // Handle remainder
            let remainder_start = chunks * 4;
            for i in remainder_start..field.len() {
                out[i] = field[i] * scalar;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn scale_field_avx2(field: &Array3<f64>, scalar: f64, out: &mut Array3<f64>) {
        if let (Some(field_slice), Some(out_slice)) = (field.as_slice(), out.as_slice_mut()) {
            // SAFETY: Feature detection ensures AVX2 is available
            unsafe { Self::scale_field_avx2_inner(field_slice, scalar, out_slice) }
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn scale_field_neon(field: &Array3<f64>, scalar: f64, out: &mut Array3<f64>) {
        use std::arch::aarch64::*;

        if let (Some(field_slice), Some(out_slice)) = (field.as_slice(), out.as_slice_mut()) {
            // SAFETY: Feature detection ensures NEON is available
            unsafe {
                let scalar_vec = vdupq_n_f64(scalar);
                let chunks = field_slice.len() / 2;

                for i in 0..chunks {
                    let offset = i * 2;
                    let v = vld1q_f64(field_slice.as_ptr().add(offset));
                    let scaled = vmulq_f64(v, scalar_vec);
                    vst1q_f64(out_slice.as_mut_ptr().add(offset), scaled);
                }

                // Handle remainder
                let remainder_start = chunks * 2;
                for i in remainder_start..field_slice.len() {
                    out_slice[i] = field_slice[i] * scalar;
                }
            }
        }
    }

    /// SWAR fallback for scaling
    fn scale_field_swar(field: &Array3<f64>, scalar: f64, out: &mut Array3<f64>) {
        const UNROLL: usize = 4;

        if let (Some(field_slice), Some(out_slice)) = (field.as_slice(), out.as_slice_mut()) {
            let chunks = field_slice.len() / UNROLL;
            let mut i = 0;

            // Unrolled loop
            for _ in 0..chunks {
                out_slice[i] = field_slice[i] * scalar;
                out_slice[i + 1] = field_slice[i + 1] * scalar;
                out_slice[i + 2] = field_slice[i + 2] * scalar;
                out_slice[i + 3] = field_slice[i + 3] * scalar;
                i += UNROLL;
            }

            // Handle remainder
            while i < field_slice.len() {
                out_slice[i] = field_slice[i] * scalar;
                i += 1;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn norm_avx2_inner(field: &[f64]) -> f64 {
        unsafe {
            use std::arch::x86_64::*;

            let mut sum_vec = _mm256_setzero_pd();
            let chunks = field.len() / 4;

            for i in 0..chunks {
                let offset = i * 4;
                let v = _mm256_loadu_pd(field.as_ptr().add(offset));
                let squared = _mm256_mul_pd(v, v);
                sum_vec = _mm256_add_pd(sum_vec, squared);
            }

            // Horizontal sum
            let mut sum_array = [0.0; 4];
            _mm256_storeu_pd(sum_array.as_mut_ptr(), sum_vec);
            let mut sum = sum_array.iter().sum::<f64>();

            // Handle remainder
            let remainder_start = chunks * 4;
            for i in remainder_start..field.len() {
                sum += field[i] * field[i];
            }

            sum.sqrt()
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn norm_avx2(field: &Array3<f64>) -> f64 {
        if let Some(field_slice) = field.as_slice() {
            // SAFETY: Feature detection ensures AVX2 is available
            unsafe { Self::norm_avx2_inner(field_slice) }
        } else {
            Self::norm_swar(field)
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn norm_neon(field: &Array3<f64>) -> f64 {
        use std::arch::aarch64::*;

        if let Some(field_slice) = field.as_slice() {
            // SAFETY: Feature detection ensures NEON is available
            unsafe {
                let mut sum_vec = vdupq_n_f64(0.0);
                let chunks = field_slice.len() / 2;

                for i in 0..chunks {
                    let offset = i * 2;
                    let v = vld1q_f64(field_slice.as_ptr().add(offset));
                    let squared = vmulq_f64(v, v);
                    sum_vec = vaddq_f64(sum_vec, squared);
                }

                // Extract sum
                let mut sum = vgetq_lane_f64(sum_vec, 0) + vgetq_lane_f64(sum_vec, 1);

                // Handle remainder
                let remainder_start = chunks * 2;
                for i in remainder_start..field_slice.len() {
                    sum += field_slice[i] * field_slice[i];
                }

                sum.sqrt()
            }
        } else {
            Self::norm_swar(field)
        }
    }

    /// SWAR fallback for norm computation
    fn norm_swar(field: &Array3<f64>) -> f64 {
        const UNROLL: usize = 4;

        if let Some(field_slice) = field.as_slice() {
            let chunks = field_slice.len() / UNROLL;
            let mut i = 0;
            let mut sum = 0.0;

            // Unrolled loop with multiple accumulators for ILP
            let mut sum0 = 0.0;
            let mut sum1 = 0.0;
            let mut sum2 = 0.0;
            let mut sum3 = 0.0;

            for _ in 0..chunks {
                sum0 += field_slice[i] * field_slice[i];
                sum1 += field_slice[i + 1] * field_slice[i + 1];
                sum2 += field_slice[i + 2] * field_slice[i + 2];
                sum3 += field_slice[i + 3] * field_slice[i + 3];
                i += UNROLL;
            }

            sum = sum0 + sum1 + sum2 + sum3;

            // Handle remainder
            while i < field_slice.len() {
                sum += field_slice[i] * field_slice[i];
                i += 1;
            }

            sum.sqrt()
        } else {
            // Fallback to iterator
            field.iter().map(|&x| x * x).sum::<f64>().sqrt()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_add_fields_correctness() {
        let a = Array3::from_elem((10, 10, 10), 1.0);
        let b = Array3::from_elem((10, 10, 10), 2.0);
        let result = SimdOps::add_fields(&a, &b);

        for &val in result.iter() {
            assert_relative_eq!(val, 3.0, epsilon = 1e-10);
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

        // sqrt(1000) ≈ 31.622776
        assert_relative_eq!(norm, (1000.0_f64).sqrt(), epsilon = 1e-10);
    }
}
