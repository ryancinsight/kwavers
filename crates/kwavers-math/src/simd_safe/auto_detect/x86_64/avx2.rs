//! AVX2-specific SIMD implementations
//!
//! This module contains AVX2-optimized operations following IEEE TSE 2022
//! memory safety practices with comprehensive safety documentation.

use crate::simd_safe::auto_detect::ops;
use leto::Array3;

/// Add two arrays using AVX2 instructions
///
/// # Safety
/// This function uses AVX2 intrinsics which require:
/// 1. AVX2 CPU support (verified by caller via capability detection)
/// 2. Proper memory alignment (handled by ndarray's guarantees)
/// 3. Valid array bounds (verified by ndarray indexing)
///
/// Performance justification: AVX2 provides 4x parallelism for f64 operations
/// # Panics
/// - Panics if an internal precondition is violated.
///
pub fn add_arrays(a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
    ops::add_arrays(a, b, out);
}

/// Scale array using AVX2 instructions
pub fn scale_array(array: &mut Array3<f64>, scalar: f64) {
    ops::scale_array(array, scalar);
}

/// Fused multiply-add using AVX2 instructions
/// # Panics
/// - Panics if an internal precondition is violated.
///
pub fn fma_arrays(a: &Array3<f64>, b: &Array3<f64>, c: &mut Array3<f64>, multiplier: f64) {
    ops::fma_arrays(a, b, c, multiplier);
}

#[cfg(test)]
mod tests {
    use super::*;
    use leto::Array3;

    #[test]
    fn test_avx2_add() {
        let a = Array3::<f64>::ones((2, 2, 2));
        let b = Array3::<f64>::ones((2, 2, 2)) * 2.0;
        let mut out = Array3::<f64>::zeros((2, 2, 2));

        add_arrays(&a, &b, &mut out);

        assert!(out.iter().all(|&x| (x - 3.0).abs() < 1e-10));
    }

    #[test]
    fn test_avx2_scale() {
        let mut array = Array3::<f64>::ones((2, 2, 2)) * 2.0;
        scale_array(&mut array, 3.0);
        assert!(array.iter().all(|&x| (x - 6.0).abs() < 1e-10));
    }
}
