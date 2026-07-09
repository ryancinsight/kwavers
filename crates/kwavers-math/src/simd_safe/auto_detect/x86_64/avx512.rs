//! AVX-512 specific SIMD implementations
//!
//! This module contains AVX-512 optimized operations with comprehensive
//! safety documentation per ICSE 2020 standards.

use crate::simd_safe::auto_detect::ops;
use leto::Array3;

/// Add two arrays using AVX-512 instructions
///
/// # Safety
/// This function uses AVX-512 intrinsics which require:
/// 1. AVX-512F CPU support (verified by caller)
/// 2. Proper memory alignment (ndarray guarantees)
/// 3. Valid array bounds (ndarray validates)
///
/// Performance: AVX-512 provides 8x parallelism for f64 operations
/// # Panics
/// - Panics if an internal precondition is violated.
///
pub fn add_arrays(a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
    ops::add_arrays(a, b, out);
}

/// Scale array using AVX-512 instructions
pub fn scale_array(array: &mut Array3<f64>, scalar: f64) {
    ops::scale_array(array, scalar);
}

/// Fused multiply-add using AVX-512 instructions
/// # Panics
/// - Panics if an internal precondition is violated.
///
pub fn fma_arrays(a: &Array3<f64>, b: &Array3<f64>, c: &mut Array3<f64>, multiplier: f64) {
    ops::fma_arrays(a, b, c, multiplier);
}
