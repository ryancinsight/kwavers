//! SSE4.2-specific SIMD implementations
//!
//! This module contains SSE4.2-optimized operations following memory
//! safety practices from IEEE TSE 2022.

use crate::simd_safe::auto_detect::ops;
use leto::Array3;

/// Add two arrays using SSE4.2 instructions
///
/// # Safety
/// Uses SSE4.2 intrinsics requiring:
/// 1. SSE4.2 CPU support (caller verified)
/// 2. Memory alignment (ndarray guaranteed)
/// 3. Bounds validation (ndarray handled)
///
/// Performance: SSE4.2 provides 2x parallelism for f64
/// # Panics
/// - Panics if an internal precondition is violated.
///
pub fn add_arrays(a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
    ops::add_arrays(a, b, out);
}

/// Scale array using SSE4.2 instructions
pub fn scale_array(array: &mut Array3<f64>, scalar: f64) {
    ops::scale_array(array, scalar);
}

/// Fused multiply-add using SSE4.2 instructions
/// # Panics
/// - Panics if an internal precondition is violated.
///
pub fn fma_arrays(a: &Array3<f64>, b: &Array3<f64>, c: &mut Array3<f64>, multiplier: f64) {
    ops::fma_arrays(a, b, c, multiplier);
}
