//! Automatic SIMD detection and dispatch
//!
//! Provides runtime CPU feature detection and automatic selection of the
//! optimal SIMD implementation for the current architecture.
//!
//! # Design Principles
//! - Zero-cost abstraction via inlining
//! - Automatic architecture detection
//! - Safe fallback to SWAR
//! - In-place operations for zero-copy

// Allow unsafe code for SIMD performance optimization
#![allow(unsafe_code)]

use ndarray::{Array3, Zip};
use std::arch::is_x86_feature_detected;

/// SIMD capability detection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimdCapability {
    /// AVX-512 available (`x86_64`)
    Avx512,
    /// AVX2 available (`x86_64`)
    Avx2,
    /// SSE4.2 available (`x86_64`)
    Sse42,
    /// NEON available (ARM)
    Neon,
    /// No SIMD, use SWAR
    Swar,
}

impl SimdCapability {
    /// Detect the best available SIMD capability
    #[inline]
    #[must_use]
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return Self::Avx512;
            }
            if is_x86_feature_detected!("avx2") {
                return Self::Avx2;
            }
            if is_x86_feature_detected!("sse4.2") {
                return Self::Sse42;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON is mandatory on AArch64
            return Self::Neon;
        }

        Self::Swar
    }

    /// Vector width in f64 elements
    #[inline]
    #[must_use]
    pub const fn vector_width(&self) -> usize {
        match self {
            Self::Avx512 => 8,
            Self::Avx2 => 4,
            Self::Sse42 => 2,
            Self::Neon => 2,
            Self::Swar => 4, // Unrolled scalar
        }
    }
}

/// Auto-dispatching SIMD operations
#[derive(Debug)]
pub struct SimdAuto {
    capability: SimdCapability,
}

impl SimdAuto {
    /// Create with automatic detection
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            capability: SimdCapability::detect(),
        }
    }

    /// Add arrays in-place: out = a + b
    #[inline]
    pub fn add_inplace(&self, a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
        debug_assert_eq!(a.shape(), b.shape());
        debug_assert_eq!(a.shape(), out.shape());

        match self.capability {
            SimdCapability::Avx512 => self.add_avx512(a, b, out),
            SimdCapability::Avx2 => self.add_avx2(a, b, out),
            SimdCapability::Sse42 => self.add_sse42(a, b, out),
            SimdCapability::Neon => self.add_neon(a, b, out),
            SimdCapability::Swar => self.add_swar(a, b, out),
        }
    }

    /// Multiply array by scalar in-place
    #[inline]
    pub fn scale_inplace(&self, array: &mut Array3<f64>, scalar: f64) {
        match self.capability {
            SimdCapability::Avx512 => self.scale_avx512(array, scalar),
            SimdCapability::Avx2 => self.scale_avx2(array, scalar),
            SimdCapability::Sse42 => self.scale_sse42(array, scalar),
            SimdCapability::Neon => self.scale_neon(array, scalar),
            SimdCapability::Swar => self.scale_swar(array, scalar),
        }
    }

    /// Fused multiply-add in-place: out = a * b + c
    #[inline]
    pub fn fma_inplace(
        &self,
        a: &Array3<f64>,
        b: &Array3<f64>,
        c: &Array3<f64>,
        out: &mut Array3<f64>,
    ) {
        debug_assert_eq!(a.shape(), b.shape());
        debug_assert_eq!(a.shape(), c.shape());
        debug_assert_eq!(a.shape(), out.shape());

        match self.capability {
            SimdCapability::Avx512 => self.fma_avx512(a, b, c, out),
            SimdCapability::Avx2 => self.fma_avx2(a, b, c, out),
            SimdCapability::Sse42 => self.fma_sse42(a, b, c, out),
            SimdCapability::Neon => self.fma_neon(a, b, c, out),
            SimdCapability::Swar => self.fma_swar(a, b, c, out),
        }
    }

    // AVX-512 implementations
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    #[inline]
    fn add_avx512(&self, a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
        use std::arch::x86_64::*;

        // Safe access to slices with fallback
        if let (Some(a_slice), Some(b_slice), Some(out_slice)) =
            (a.as_slice(), b.as_slice(), out.as_slice_mut())
        {
            let chunks = a_slice.len() / 8;
            let remainder = a_slice.len() % 8;

            // SAFETY: AVX-512 is available (checked in detect())
            unsafe {
                for i in 0..chunks {
                    let idx = i * 8;
                    let va = _mm512_loadu_pd(&a_slice[idx]);
                    let vb = _mm512_loadu_pd(&b_slice[idx]);
                    let vr = _mm512_add_pd(va, vb);
                    _mm512_storeu_pd(&mut out_slice[idx], vr);
                }
            }

            // Handle remainder
            let remainder_start = chunks * 8;
            for i in remainder_start..a_slice.len() {
                out_slice[i] = a_slice[i] + b_slice[i];
            }
        } else {
            // Fallback for non-contiguous arrays
            self.add_swar(a, b, out);
        }
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    #[inline]
    fn add_avx512(&self, a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
        self.add_swar(a, b, out); // Fallback
    }

    // AVX2 implementations
    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn add_avx2(&self, a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
        if is_x86_feature_detected!("avx2") {
            use std::arch::x86_64::{_mm256_add_pd, _mm256_loadu_pd, _mm256_storeu_pd};

            let a_slice = a.as_slice().unwrap();
            let b_slice = b.as_slice().unwrap();
            let out_slice = out.as_slice_mut().unwrap();

            let chunks = a_slice.len() / 4;
            let _remainder = a_slice.len() % 4;

            // SAFETY: AVX2 availability verified by is_x86_feature_detected!
            // Array bounds checked: idx ranges from 0 to chunks*4, where chunks = len/4
            // Slice pointers are valid as they come from Array3::as_slice operations
            // Memory is properly aligned for AVX2 load/store operations
            unsafe {
                for i in 0..chunks {
                    let idx = i * 4;
                    let va = _mm256_loadu_pd(&a_slice[idx]);
                    let vb = _mm256_loadu_pd(&b_slice[idx]);
                    let vr = _mm256_add_pd(va, vb);
                    _mm256_storeu_pd(&mut out_slice[idx], vr);
                }
            }

            // Handle remainder
            let remainder_start = chunks * 4;
            for i in remainder_start..a_slice.len() {
                out_slice[i] = a_slice[i] + b_slice[i];
            }
        } else {
            self.add_swar(a, b, out);
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    #[inline]
    fn add_avx2(&self, a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
        self.add_swar(a, b, out);
    }

    // SSE4.2 implementations
    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn add_sse42(&self, a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
        if is_x86_feature_detected!("sse4.2") {
            use std::arch::x86_64::{_mm_add_pd, _mm_loadu_pd, _mm_storeu_pd};

            let a_slice = a.as_slice().unwrap();
            let b_slice = b.as_slice().unwrap();
            let out_slice = out.as_slice_mut().unwrap();

            let chunks = a_slice.len() / 2;
            let remainder = a_slice.len() % 2;

            // SAFETY: SSE4.2 availability verified by is_x86_feature_detected!
            // Array bounds checked: idx ranges from 0 to chunks*2, where chunks = len/2
            // Slice pointers are valid as they come from Array3::as_slice operations
            // Memory is properly aligned for SSE load/store operations
            unsafe {
                for i in 0..chunks {
                    let idx = i * 2;
                    let va = _mm_loadu_pd(&a_slice[idx]);
                    let vb = _mm_loadu_pd(&b_slice[idx]);
                    let vr = _mm_add_pd(va, vb);
                    _mm_storeu_pd(&mut out_slice[idx], vr);
                }
            }

            // Handle remainder
            if remainder > 0 {
                let idx = chunks * 2;
                out_slice[idx] = a_slice[idx] + b_slice[idx];
            }
        } else {
            self.add_swar(a, b, out);
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    #[inline]
    fn add_sse42(&self, a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
        self.add_swar(a, b, out);
    }

    // NEON implementations
    #[cfg(target_arch = "aarch64")]
    #[inline]
    fn add_neon(&self, a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
        use std::arch::aarch64::*;

        let a_slice = a.as_slice().unwrap();
        let b_slice = b.as_slice().unwrap();
        let out_slice = out.as_slice_mut().unwrap();

        let chunks = a_slice.len() / 2;
        let remainder = a_slice.len() % 2;

        // SAFETY: NEON is mandatory on AArch64
        unsafe {
            for i in 0..chunks {
                let idx = i * 2;
                let va = vld1q_f64(&a_slice[idx]);
                let vb = vld1q_f64(&b_slice[idx]);
                let vr = vaddq_f64(va, vb);
                vst1q_f64(&mut out_slice[idx], vr);
            }
        }

        // Handle remainder
        if remainder > 0 {
            let idx = chunks * 2;
            out_slice[idx] = a_slice[idx] + b_slice[idx];
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    #[inline]
    fn add_neon(&self, a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
        self.add_swar(a, b, out);
    }

    // SWAR (SIMD Within A Register) fallback
    #[inline]
    fn add_swar(&self, a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
        // Use ndarray's optimized iterators with unrolling
        Zip::from(out.view_mut())
            .and(a.view())
            .and(b.view())
            .for_each(|o, &a, &b| *o = a + b);
    }

    // Scale implementations follow similar pattern
    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn scale_avx2(&self, array: &mut Array3<f64>, scalar: f64) {
        if is_x86_feature_detected!("avx2") {
            use std::arch::x86_64::{
                _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_storeu_pd,
            };

            let slice = array.as_slice_mut().unwrap();
            let chunks = slice.len() / 4;
            let _remainder = slice.len() % 4;

            // SAFETY: AVX2 is available (runtime check)
            unsafe {
                let vs = _mm256_set1_pd(scalar);
                for i in 0..chunks {
                    let idx = i * 4;
                    let va = _mm256_loadu_pd(&slice[idx]);
                    let vr = _mm256_mul_pd(va, vs);
                    _mm256_storeu_pd(&mut slice[idx], vr);
                }
            }

            // Handle remainder
            let remainder_start = chunks * 4;
            for i in remainder_start..slice.len() {
                slice[i] *= scalar;
            }
        } else {
            self.scale_swar(array, scalar);
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    #[inline]
    fn scale_avx2(&self, array: &mut Array3<f64>, scalar: f64) {
        self.scale_swar(array, scalar);
    }

    // Stub implementations for other scale variants
    #[inline]
    fn scale_avx512(&self, array: &mut Array3<f64>, scalar: f64) {
        self.scale_avx2(array, scalar); // Delegate to AVX2 for now
    }

    #[inline]
    fn scale_sse42(&self, array: &mut Array3<f64>, scalar: f64) {
        self.scale_swar(array, scalar);
    }

    #[inline]
    fn scale_neon(&self, array: &mut Array3<f64>, scalar: f64) {
        self.scale_swar(array, scalar);
    }

    #[inline]
    fn scale_swar(&self, array: &mut Array3<f64>, scalar: f64) {
        array.mapv_inplace(|x| x * scalar);
    }

    // FMA implementations
    #[inline]
    fn fma_avx512(&self, a: &Array3<f64>, b: &Array3<f64>, c: &Array3<f64>, out: &mut Array3<f64>) {
        self.fma_swar(a, b, c, out); // Delegate for now
    }

    #[inline]
    fn fma_avx2(&self, a: &Array3<f64>, b: &Array3<f64>, c: &Array3<f64>, out: &mut Array3<f64>) {
        self.fma_swar(a, b, c, out);
    }

    #[inline]
    fn fma_sse42(&self, a: &Array3<f64>, b: &Array3<f64>, c: &Array3<f64>, out: &mut Array3<f64>) {
        self.fma_swar(a, b, c, out);
    }

    #[inline]
    fn fma_neon(&self, a: &Array3<f64>, b: &Array3<f64>, c: &Array3<f64>, out: &mut Array3<f64>) {
        self.fma_swar(a, b, c, out);
    }

    #[inline]
    fn fma_swar(&self, a: &Array3<f64>, b: &Array3<f64>, c: &Array3<f64>, out: &mut Array3<f64>) {
        Zip::from(out.view_mut())
            .and(a.view())
            .and(b.view())
            .and(c.view())
            .for_each(|o, &a, &b, &c| *o = a.mul_add(b, c));
    }
}

impl Default for SimdAuto {
    fn default() -> Self {
        Self::new()
    }
}

// Global SIMD instance with lazy initialization
lazy_static::lazy_static! {
    static ref SIMD: SimdAuto = SimdAuto::new();
}

/// Get the global SIMD processor
#[inline]
#[must_use]
pub fn simd() -> &'static SimdAuto {
    &SIMD
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_detection() {
        let capability = SimdCapability::detect();
        println!("Detected SIMD capability: {:?}", capability);
        assert!(capability.vector_width() > 0);
    }

    #[test]
    fn test_add_inplace() {
        let simd = SimdAuto::new();
        let a = Array3::from_elem((4, 4, 4), 1.0);
        let b = Array3::from_elem((4, 4, 4), 2.0);
        let mut out = Array3::zeros((4, 4, 4));

        simd.add_inplace(&a, &b, &mut out);

        for &val in out.iter() {
            assert!((val - 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_scale_inplace() {
        let simd = SimdAuto::new();
        let mut array = Array3::from_elem((4, 4, 4), 2.0);

        simd.scale_inplace(&mut array, 3.0);

        for &val in array.iter() {
            assert!((val - 6.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_fma_inplace() {
        let simd = SimdAuto::new();
        let a = Array3::from_elem((4, 4, 4), 2.0);
        let b = Array3::from_elem((4, 4, 4), 3.0);
        let c = Array3::from_elem((4, 4, 4), 1.0);
        let mut out = Array3::zeros((4, 4, 4));

        simd.fma_inplace(&a, &b, &c, &mut out);

        for &val in out.iter() {
            assert!((val - 7.0).abs() < 1e-10); // 2 * 3 + 1 = 7
        }
    }
}
