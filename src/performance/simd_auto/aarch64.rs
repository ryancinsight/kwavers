//! ARM NEON SIMD implementation for aarch64
//!
//! Provides NEON-optimized operations for ARM64 architecture.
//! Reference: ARM NEON Programmer's Guide

pub mod neon {
    //! NEON intrinsics-based implementations

    /// Add two arrays element-wise using NEON
    #[inline]
    pub unsafe fn add_arrays(a: &[f64], b: &[f64], out: &mut [f64]) {
        // Fallback to scalar for now - NEON f64 support is limited
        for i in 0..a.len() {
            out[i] = a[i] + b[i];
        }
    }

    /// Scale array by scalar using NEON
    #[inline]
    pub unsafe fn scale_array(array: &mut [f64], scalar: f64) {
        // Fallback to scalar for now
        for value in array.iter_mut() {
            *value *= scalar;
        }
    }

    /// Fused multiply-add: out[i] = a[i] * b[i] + c[i] * multiplier
    #[inline]
    pub unsafe fn fma_arrays(a: &[f64], b: &[f64], c: &[f64], multiplier: f64) -> Vec<f64> {
        // Fallback to scalar for now
        a.iter()
            .zip(b.iter())
            .zip(c.iter())
            .map(|((&ai, &bi), &ci)| ai * bi + ci * multiplier)
            .collect()
    }
}
