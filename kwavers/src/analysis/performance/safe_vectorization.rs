//! Zero-cost safe vectorization using iterator combinators
//!
//! Replaces unsafe SIMD intrinsics with portable iterator patterns that enable
//! LLVM autovectorization without architecture-specific unsafe code.
//!
//! Performance characteristics:
//! - LLVM auto-vectorization with -O2/O3 flags
//! - Zero unsafe code blocks
//! - Portable across all architectures
//! - Rayon parallelization for large arrays

use ndarray::Array3;

/// Safe vectorization operations using iterator combinators
#[derive(Debug, Clone, Copy)]
pub struct SafeVectorOps;

impl SafeVectorOps {
    /// Add two arrays element-wise.
    ///
    /// Uses ndarray's native `+` operator which LLVM autovectorizes to AVX2/NEON
    /// when available. No intermediate Vec allocation.
    #[inline]
    #[must_use]
    pub fn add_arrays(a: &Array3<f64>, b: &Array3<f64>) -> Array3<f64> {
        debug_assert_eq!(a.dim(), b.dim(), "Array dimensions must match");
        a + b
    }

    /// Parallel add for large arrays using Rayon + ndarray Zip (no intermediate Vec)
    #[inline]
    #[must_use]
    pub fn add_arrays_parallel(a: &Array3<f64>, b: &Array3<f64>) -> Array3<f64> {
        debug_assert_eq!(a.dim(), b.dim(), "Array dimensions must match");
        let mut result = Array3::<f64>::zeros(a.dim());
        ndarray::Zip::from(&mut result)
            .and(a)
            .and(b)
            .par_for_each(|r, &av, &bv| *r = av + bv);
        result
    }

    /// Scalar multiplication using ndarray mapv (no intermediate Vec allocation).
    #[inline]
    #[must_use]
    pub fn scalar_multiply(array: &Array3<f64>, scalar: f64) -> Array3<f64> {
        array.mapv(|v| v * scalar)
    }

    /// In-place scalar multiplication for zero-copy operations
    #[inline]
    pub fn scalar_multiply_inplace(array: &mut Array3<f64>, scalar: f64) {
        array.iter_mut().for_each(|val| *val *= scalar);
    }

    /// Element-wise exponential using ndarray mapv (no intermediate Vec allocation).
    #[inline]
    #[must_use]
    pub fn exp_array(array: &Array3<f64>) -> Array3<f64> {
        array.mapv(f64::exp)
    }

    /// Dot product using fold for LLVM reduction optimization
    #[inline]
    #[must_use]
    pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len(), "Slice lengths must match");

        a.iter()
            .zip(b.iter())
            .fold(0.0, |acc, (a_val, b_val)| acc + a_val * b_val)
    }

    /// Array L2 norm using iterator chain
    #[inline]
    #[must_use]
    pub fn l2_norm(array: &Array3<f64>) -> f64 {
        array
            .iter()
            .map(|val| val * val)
            .fold(0.0, |acc, val_sq| acc + val_sq)
            .sqrt()
    }

    /// Chunked operations for cache optimization.
    ///
    /// Processes contiguous input in `chunk_size`-element cache-friendly strips.
    /// Falls back to element-wise ndarray Zip for non-contiguous arrays.
    /// The `chunk_size` parameter is advisory; the contiguous fast-path still uses
    /// it for L1-cache tiling; the fallback ignores it (ndarray handles tiling).
    #[inline]
    #[must_use]
    pub fn add_arrays_chunked(a: &Array3<f64>, b: &Array3<f64>, chunk_size: usize) -> Array3<f64> {
        debug_assert_eq!(a.dim(), b.dim(), "Array dimensions must match");

        let mut result = Array3::<f64>::zeros(a.dim());

        if let (Some(a_slice), Some(b_slice), Some(r_slice)) =
            (a.as_slice(), b.as_slice(), result.as_slice_mut())
        {
            // Contiguous fast-path: iterate chunks for cache-line tiling
            for ((r_chunk, a_chunk), b_chunk) in r_slice
                .chunks_mut(chunk_size)
                .zip(a_slice.chunks(chunk_size))
                .zip(b_slice.chunks(chunk_size))
            {
                for ((r, &av), &bv) in r_chunk.iter_mut().zip(a_chunk).zip(b_chunk) {
                    *r = av + bv;
                }
            }
        } else {
            // Non-contiguous fallback — ndarray Zip handles arbitrary strides
            ndarray::Zip::from(&mut result)
                .and(a)
                .and(b)
                .for_each(|r, &av, &bv| *r = av + bv);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_add_arrays_correctness() {
        let a = Array3::from_elem((2, 2, 2), 1.0);
        let b = Array3::from_elem((2, 2, 2), 2.0);
        let result = SafeVectorOps::add_arrays(&a, &b);

        assert_relative_eq!(result[[0, 0, 0]], 3.0);
        assert_relative_eq!(result[[1, 1, 1]], 3.0);
    }

    #[test]
    fn test_scalar_multiply_correctness() {
        let a = Array3::from_elem((2, 2, 2), 2.0);
        let result = SafeVectorOps::scalar_multiply(&a, 3.0);

        assert_relative_eq!(result[[0, 0, 0]], 6.0);
        assert_relative_eq!(result[[1, 1, 1]], 6.0);
    }

    #[test]
    fn test_dot_product_correctness() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = SafeVectorOps::dot_product(&a, &b);

        assert_relative_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_l2_norm_correctness() {
        let a = Array3::from_elem((2, 2, 2), 1.0);
        let result = SafeVectorOps::l2_norm(&a);

        assert_relative_eq!(result, (8.0_f64).sqrt()); // sqrt(8 * 1^2)
    }
}
