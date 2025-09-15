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
use rayon::prelude::*;

/// Safe vectorization operations using iterator combinators
#[derive(Debug, Clone, Copy)]
pub struct SafeVectorOps;

impl SafeVectorOps {
    /// Add two arrays element-wise using safe iterator patterns
    ///
    /// LLVM will autovectorize this to use AVX2/NEON instructions when available.
    /// No unsafe code required - the compiler handles vectorization optimization.
    #[inline]
    #[must_use]
    pub fn add_arrays(a: &Array3<f64>, b: &Array3<f64>) -> Array3<f64> {
        debug_assert_eq!(a.dim(), b.dim(), "Array dimensions must match");

        // Use zip iterator for element-wise operations
        // LLVM will vectorize this pattern automatically
        let result: Vec<f64> = a
            .iter()
            .zip(b.iter())
            .map(|(a_val, b_val)| a_val + b_val)
            .collect();

        Array3::from_shape_vec(a.dim(), result).expect("Shape and data length must match")
    }

    /// Parallel add for large arrays using Rayon
    #[inline]
    #[must_use]
    pub fn add_arrays_parallel(a: &Array3<f64>, b: &Array3<f64>) -> Array3<f64> {
        debug_assert_eq!(a.dim(), b.dim(), "Array dimensions must match");

        // Convert to vectors first, then process in parallel
        let a_vec: Vec<f64> = a.iter().cloned().collect();
        let b_vec: Vec<f64> = b.iter().cloned().collect();

        let result: Vec<f64> = a_vec
            .par_iter()
            .zip(b_vec.par_iter())
            .map(|(a_val, b_val)| a_val + b_val)
            .collect();

        Array3::from_shape_vec(a.dim(), result).expect("Shape and data length must match")
    }

    /// Scalar multiplication using iterator patterns
    #[inline]
    #[must_use]
    pub fn scalar_multiply(array: &Array3<f64>, scalar: f64) -> Array3<f64> {
        let result: Vec<f64> = array.iter().map(|val| val * scalar).collect();

        Array3::from_shape_vec(array.dim(), result).expect("Shape and data length must match")
    }

    /// In-place scalar multiplication for zero-copy operations
    #[inline]
    pub fn scalar_multiply_inplace(array: &mut Array3<f64>, scalar: f64) {
        array.iter_mut().for_each(|val| *val *= scalar);
    }

    /// Element-wise exponential with safe iterator pattern
    #[inline]
    #[must_use]
    pub fn exp_array(array: &Array3<f64>) -> Array3<f64> {
        let result: Vec<f64> = array.iter().map(|val| val.exp()).collect();

        Array3::from_shape_vec(array.dim(), result).expect("Shape and data length must match")
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

    /// Chunked operations for cache optimization
    #[inline]
    #[must_use]
    pub fn add_arrays_chunked(a: &Array3<f64>, b: &Array3<f64>, chunk_size: usize) -> Array3<f64> {
        debug_assert_eq!(a.dim(), b.dim(), "Array dimensions must match");

        // Use safe iteration for non-contiguous arrays
        if let (Some(a_slice), Some(b_slice)) = (a.as_slice(), b.as_slice()) {
            let result: Vec<f64> = a_slice
                .chunks(chunk_size)
                .zip(b_slice.chunks(chunk_size))
                .flat_map(|(a_chunk, b_chunk)| {
                    a_chunk
                        .iter()
                        .zip(b_chunk.iter())
                        .map(|(a_val, b_val)| a_val + b_val)
                })
                .collect();

            Array3::from_shape_vec(a.dim(), result).expect("Shape and data length must match")
        } else {
            // Fallback for non-contiguous arrays
            let mut result = Array3::zeros(a.dim());
            result.iter_mut().zip(a.iter()).zip(b.iter()).for_each(|((out, &a_val), &b_val)| {
                *out = a_val + b_val;
            });
            result
        }
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
