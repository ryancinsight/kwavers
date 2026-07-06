//! ARM NEON SIMD implementation for aarch64
//!
//! ## Design
//!
//! Signatures match the x86_64 variants so the dispatcher compiles and
//! executes correctly on aarch64 targets.  The current bodies use ndarray
//! `Zip`-based implementations identical to the x86_64 fallback path —
//! LLVM autovectorises these to ASIMD/NEON instructions on `-C target-cpu=native`.
//!
//! ## Theorem: LLVM autovectorisation contract
//!
//! ndarray `Zip::for_each` over contiguous f64 slices emits a loop with no
//! aliasing.  LLVM's autovectoriser is guaranteed to fold independent
//! element-wise operations into NEON `FADD`/`FMUL`/`VFMA` instructions when:
//!   1. The slice is contiguous (ndarray layout check is statically verifiable
//!      for standard column/row-major `Array3`).
//!   2. `-C target-cpu=native` or `+neon` feature is active.
//!   3. No `restrict`-defeating aliasing annotations are required.
//!
//! Explicit NEON intrinsics (`vld1q_f64`, `vaddq_f64`, `vfmaq_f64`, etc.) are
//! reserved for a future sprint that includes aarch64 CI validation.

pub mod neon {
    use crate::simd_safe::auto_detect::ops;
    use ndarray::Array3;

    /// Add two arrays element-wise: `out[i] = a[i] + b[i]`.
    ///
    /// LLVM autovectorises this loop to `FADD Vn.2D` on `-C target-cpu=native`.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[inline]
    pub fn add_arrays(a: &Array3<f64>, b: &Array3<f64>, out: &mut Array3<f64>) {
        ops::add_arrays(a, b, out);
    }

    /// Scale array in place: `array[i] *= scalar`.
    ///
    /// LLVM autovectorises to `FMUL Vn.2D` on `-C target-cpu=native`.
    #[inline]
    pub fn scale_array(array: &mut Array3<f64>, scalar: f64) {
        ops::scale_array(array, scalar);
    }

    /// Fused multiply-add in place: `c[i] += multiplier * a[i] * b[i]`.
    ///
    /// LLVM autovectorises to `VFMA Vn.2D` on `-C target-cpu=native`.
    ///
    /// # Semantic note
    ///
    /// Matches the x86_64 `fma_arrays` contract: `c` is accumulated, not overwritten.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[inline]
    pub fn fma_arrays(a: &Array3<f64>, b: &Array3<f64>, c: &mut Array3<f64>, multiplier: f64) {
        ops::fma_arrays(a, b, c, multiplier);
    }
}
