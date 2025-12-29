//! Main SIMD operations structure with architecture dispatch

use ndarray::Array3;

/// SIMD lane width for f64 operations
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
const SIMD_WIDTH: usize = 4; // AVX2: 256 bits / 64 bits = 4

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
const SIMD_WIDTH: usize = 2; // NEON: 128 bits / 64 bits = 2

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[allow(dead_code)]
const SIMD_WIDTH: usize = 1; // Scalar fallback

/// Portable SIMD operations
#[derive(Debug)]
pub struct SimdOps;

impl SimdOps {
    /// Add two fields element-wise using SIMD
    #[inline]
    #[must_use]
    pub fn add_fields(a: &Array3<f64>, b: &Array3<f64>) -> Array3<f64> {
        let shape = a.dim();
        let mut result = Array3::zeros(shape);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                super::avx2::add_fields_avx2(a, b, &mut result);
                return result;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                super::neon::add_fields_neon(a, b, &mut result);
                return result;
            }
        }

        // Fallback to SWAR
        super::swar::add_fields_swar(a, b, &mut result);
        result
    }

    /// Scale field by scalar using SIMD
    #[inline]
    #[must_use]
    pub fn scale_field(field: &Array3<f64>, scalar: f64) -> Array3<f64> {
        let shape = field.dim();
        let mut result = Array3::zeros(shape);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                super::avx2::scale_field_avx2(field, scalar, &mut result);
                return result;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                super::neon::scale_field_neon(field, scalar, &mut result);
                return result;
            }
        }

        // Fallback to SWAR
        super::swar::scale_field_swar(field, scalar, &mut result);
        result
    }

    /// Compute L2 norm using SIMD
    #[inline]
    #[must_use]
    pub fn norm(field: &Array3<f64>) -> f64 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return super::avx2::norm_avx2(field);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return super::neon::norm_neon(field);
            }
        }

        // Fallback to SWAR
        super::swar::norm_swar(field)
    }

    /// Multiply two fields element-wise using SIMD
    #[inline]
    #[must_use]
    pub fn multiply_fields(a: &Array3<f64>, b: &Array3<f64>) -> Array3<f64> {
        let shape = a.dim();
        let mut result = Array3::zeros(shape);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 feature is detected and input arrays have compatible shapes.
                #[allow(unsafe_code)]
                unsafe {
                    super::avx2::multiply_fields_avx2(a, b, &mut result);
                }
                return result;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                super::neon::multiply_fields_neon(a, b, &mut result);
                return result;
            }
        }

        // Fallback to SWAR
        super::swar::multiply_fields_swar(a, b, &mut result);
        result
    }

    /// Subtract two fields element-wise using SIMD
    #[inline]
    #[must_use]
    pub fn subtract_fields(a: &Array3<f64>, b: &Array3<f64>) -> Array3<f64> {
        let shape = a.dim();
        let mut result = Array3::zeros(shape);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 feature is detected and input arrays have compatible shapes.
                #[allow(unsafe_code)]
                unsafe {
                    super::avx2::subtract_fields_avx2(a, b, &mut result);
                }
                return result;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                super::neon::subtract_fields_neon(a, b, &mut result);
                return result;
            }
        }

        // Fallback to SWAR
        super::swar::subtract_fields_swar(a, b, &mut result);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

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

        // sqrt(1000) ≈ 31.622776
        assert_relative_eq!(norm, (1000.0_f64).sqrt(), epsilon = 1e-10);
    }
}
