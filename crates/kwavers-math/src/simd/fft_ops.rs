//! SIMD-accelerated FFT complex multiplication kernels.

use super::config::{MathSimdLevel, SimdConfig};

/// SIMD-accelerated FFT operations
#[derive(Debug)]
pub struct FftSimdOps {
    config: SimdConfig,
}

impl FftSimdOps {
    /// Create new FFT SIMD operations
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: SimdConfig::detect(),
        }
    }

    /// SIMD-accelerated complex multiplication for FFT
    pub fn complex_multiply(
        &self,
        real1: &mut [f32],
        imag1: &mut [f32],
        real2: &[f32],
        imag2: &[f32],
    ) {
        match self.config.level {
            #[cfg(target_arch = "x86_64")]
            MathSimdLevel::Avx2 => {
                // SAFETY: AVX2 intrinsics are safe here because:
                // 1. CPU feature detection ensures AVX2 availability
                // 2. Input slices are checked for compatible lengths
                // 3. Memory is properly aligned for SIMD operations
                #[allow(unsafe_code)]
                unsafe {
                    self.complex_multiply_avx2(real1, imag1, real2, imag2);
                }
            }
            _ => self.complex_multiply_scalar(real1, imag1, real2, imag2),
        }
    }

    /// Scalar complex multiplication
    fn complex_multiply_scalar(
        &self,
        real1: &mut [f32],
        imag1: &mut [f32],
        real2: &[f32],
        imag2: &[f32],
    ) {
        for i in 0..real1.len() {
            let r1 = real1[i];
            let i1 = imag1[i];
            let r2 = real2[i];
            let i2 = imag2[i];

            real1[i] = r1.mul_add(r2, -(i1 * i2));
            imag1[i] = r1.mul_add(i2, i1 * r2);
        }
    }

    /// AVX2 complex multiplication
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    /// SAFETY: Caller must ensure:
    /// - CPU supports AVX2 (verified via SimdConfig::detect)
    /// - All input slices have equal length
    /// - Memory alignment is suitable for AVX2 operations
    #[allow(unsafe_code)]
    unsafe fn complex_multiply_avx2(
        &self,
        real1: &mut [f32],
        imag1: &mut [f32],
        real2: &[f32],
        imag2: &[f32],
    ) {
        use std::arch::x86_64::{
            _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_storeu_ps, _mm256_sub_ps,
        };

        let len = real1
            .len()
            .min(real2.len())
            .min(imag1.len())
            .min(imag2.len());
        let mut i = 0;

        while i + 7 < len {
            let r1 = _mm256_loadu_ps(real1.as_ptr().add(i));
            let i1 = _mm256_loadu_ps(imag1.as_ptr().add(i));
            let r2 = _mm256_loadu_ps(real2.as_ptr().add(i));
            let i2 = _mm256_loadu_ps(imag2.as_ptr().add(i));

            // Compute: (r1 + i*i1) * (r2 + i*i2)
            // Real part: r1*r2 - i1*i2
            // Imag part: r1*i2 + i1*r2

            let real_result = _mm256_sub_ps(_mm256_mul_ps(r1, r2), _mm256_mul_ps(i1, i2));

            let imag_result = _mm256_add_ps(_mm256_mul_ps(r1, i2), _mm256_mul_ps(i1, r2));

            _mm256_storeu_ps(real1.as_mut_ptr().add(i), real_result);
            _mm256_storeu_ps(imag1.as_mut_ptr().add(i), imag_result);

            i += 8;
        }

        // Handle remaining elements with scalar operations
        while i < len {
            let r1 = real1[i];
            let i1 = imag1[i];
            let r2 = real2[i];
            let i2 = imag2[i];

            real1[i] = r1.mul_add(r2, -(i1 * i2));
            imag1[i] = r1.mul_add(i2, i1 * r2);
            i += 1;
        }
    }
}

impl Default for FftSimdOps {
    fn default() -> Self {
        Self::new()
    }
}
