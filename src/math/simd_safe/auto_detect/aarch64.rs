//! ARM NEON SIMD implementation for aarch64
//!
//! Provides NEON-optimized operations for ARM64 architecture.
//! Reference: ARM NEON Programmer's Guide

pub mod neon {
    //! NEON intrinsics-based implementations

    /// Add two arrays element-wise using NEON
    #[inline]
    pub unsafe fn add_arrays(a: &[f64], b: &[f64], out: &mut [f64]) {
        // SAFETY: Scalar fallback implementation with unsafe signature for API compatibility
        //   - No actual SIMD intrinsics used (conservative fallback to safe scalar operations)
        //   - Pointer access via safe slice indexing only (bounds checked by Rust)
        //   - Marked unsafe to match SIMD trait requirements and maintain consistent API
        //   - Precondition: Slice lengths must satisfy a.len() <= out.len() (caller responsibility)
        // INVARIANTS:
        //   - Precondition: a.len() == b.len() and both <= out.len() (validated by caller)
        //   - Loop invariant: ∀i ∈ [0, a.len()): i < a.len() and i < b.len() and i < out.len()
        //   - Postcondition: ∀k ∈ [0, a.len()): out[k] = a[k] + b[k]
        //   - Note: Does not modify out[a.len()..] if out.len() > a.len()
        // ALTERNATIVES:
        //   - Full NEON f64 implementation using vld1q_f64/vaddq_f64/vst1q_f64
        //     (see src/math/simd_safe/neon.rs for production implementation)
        //   - Current approach: Conservative fallback ensuring correctness over performance
        //   - Reason: Development/testing on x86_64 hosts without ARM64 hardware
        //   - TODO: Replace with proper NEON intrinsics in future sprint (requires ARM64 CI/testing)
        // PERFORMANCE:
        //   - No SIMD acceleration in current implementation (scalar baseline performance)
        //   - Expected speedup: 1.0x (no optimization, baseline for correctness)
        //   - Acceptable for: Development/testing on non-ARM64 platforms
        //   - Production: Use src/math/simd_safe/neon.rs on aarch64 targets
        // Fallback to scalar for now - NEON f64 support is limited
        for i in 0..a.len() {
            out[i] = a[i] + b[i];
        }
    }

    /// Scale array by scalar using NEON
    #[inline]
    pub unsafe fn scale_array(array: &mut [f64], scalar: f64) {
        // SAFETY: Scalar fallback implementation with unsafe signature for API compatibility
        //   - No SIMD intrinsics used (safe scalar operations only)
        //   - Memory access via safe iterator (iter_mut() provides exclusive mutable access)
        //   - Marked unsafe to match SIMD trait requirements
        //   - No pointer arithmetic or raw memory access
        // INVARIANTS:
        //   - Precondition: array is a valid mutable slice (enforced by Rust borrow checker)
        //   - Postcondition: ∀k ∈ [0, array.len()): array[k] = array_old[k] × scalar
        //   - Numerical stability: Scalar multiplication is exact (no accumulation error)
        //   - Special values: NaN/Inf propagate according to IEEE-754
        // ALTERNATIVES:
        //   - Full NEON implementation using vdupq_n_f64/vmulq_f64
        //     (see src/math/simd_safe/neon.rs for production implementation)
        //   - Current approach: Conservative fallback for cross-platform development
        //   - TODO: Implement proper NEON intrinsics when ARM64 CI becomes available
        // PERFORMANCE:
        //   - No SIMD acceleration (scalar baseline)
        //   - Expected speedup: 1.0x (no optimization)
        //   - Production: Use src/math/simd_safe/neon.rs on aarch64 targets (1.8-2x speedup)
        // Fallback to scalar for now
        for value in array.iter_mut() {
            *value *= scalar;
        }
    }

    /// Fused multiply-add: out[i] = a[i] * b[i] + c[i] * multiplier
    #[inline]
    pub unsafe fn fma_arrays(a: &[f64], b: &[f64], c: &[f64], multiplier: f64) -> Vec<f64> {
        // SAFETY: Scalar fallback implementation with unsafe signature for API compatibility
        //   - No SIMD intrinsics used (safe functional iterator operations)
        //   - Memory access via safe iterators (zip provides synchronized iteration)
        //   - Marked unsafe to match SIMD trait requirements
        //   - Heap allocation for result vector (standard Rust semantics)
        // INVARIANTS:
        //   - Precondition: a, b, c have compatible lengths (zip stops at shortest)
        //   - Length: result.len() = min(a.len(), b.len(), c.len())
        //   - Postcondition: ∀k ∈ [0, result.len()): result[k] = a[k] × b[k] + c[k] × multiplier
        //   - Numerical: Fused multiply-add semantics (two operations, not FMA instruction)
        //   - Note: Not a true hardware FMA (no rounding error reduction)
        // ALTERNATIVES:
        //   - Full NEON implementation using vfmaq_f64 (fused multiply-add instruction)
        //     Hardware FMA provides better numerical accuracy (single rounding instead of two)
        //   - Current approach: Scalar fallback with separate multiply and add operations
        //   - TODO: Implement NEON vfmaq_f64 for true FMA semantics and 1.8-2x speedup
        // PERFORMANCE:
        //   - No SIMD acceleration (scalar baseline with iterator overhead)
        //   - Expected speedup: 1.0x (no optimization, possible iterator overhead)
        //   - Memory: Heap allocation for result vector (consider pre-allocated output in production)
        //   - Production: Use NEON vfmaq_f64 on aarch64 for true FMA and performance
        // Fallback to scalar for now
        a.iter()
            .zip(b.iter())
            .zip(c.iter())
            .map(|((&ai, &bi), &ci)| ai * bi + ci * multiplier)
            .collect()
    }
}
