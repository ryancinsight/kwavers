//! Vector Norms and Basic Operations
//!
//! This module provides vector norms and basic vector operations
//! commonly used in numerical algorithms and signal processing.

use ndarray::Array3;

/// Vector norms and basic operations
#[derive(Debug)]
pub struct VectorOperations;

impl VectorOperations {
    /// Compute L2 norm of a 3D array (Frobenius norm)
    ///
    /// # Arguments
    /// * `array` - 3D array of real values
    ///
    /// # Returns
    /// L2 norm (square root of sum of squares)
    pub fn norm_l2(array: &Array3<f64>) -> f64 {
        let mut sum_sq = 0.0;
        for &val in array.iter() {
            sum_sq += val * val;
        }
        sum_sq.sqrt()
    }

    /// Compute L2 norm of a 3D array slice (Frobenius norm)
    ///
    /// # Arguments
    /// * `array` - 3D array of real values
    ///
    /// # Returns
    /// L2 norm (square root of sum of squares)
    pub fn norm_l2_3d(array: &Array3<f64>) -> f64 {
        Self::norm_l2(array)
    }

    /// Normalize a vector in-place (L2 normalization)
    ///
    /// # Arguments
    /// * `vector` - Mutable vector to normalize
    ///
    /// # Returns
    /// true if normalization succeeded, false if vector is zero-length
    pub fn normalize_vector(vector: &mut [f64]) -> bool {
        let norm = Self::vector_norm_l2(vector);
        if norm > 0.0 {
            for x in vector.iter_mut() {
                *x /= norm;
            }
            true
        } else {
            false
        }
    }

    /// Compute L2 norm of a vector
    ///
    /// # Arguments
    /// * `vector` - Vector of real values
    ///
    /// # Returns
    /// L2 norm (Euclidean norm)
    pub fn vector_norm_l2(vector: &[f64]) -> f64 {
        let mut sum_sq = 0.0;
        for &x in vector {
            sum_sq += x * x;
        }
        sum_sq.sqrt()
    }

    /// Compute dot product of two vectors
    ///
    /// # Arguments
    /// * `a` - First vector
    /// * `b` - Second vector
    ///
    /// # Returns
    /// Dot product, or None if vectors have different lengths
    pub fn dot_product(a: &[f64], b: &[f64]) -> Option<f64> {
        if a.len() != b.len() {
            return None;
        }

        let mut sum = 0.0;
        for (&x, &y) in a.iter().zip(b.iter()) {
            sum += x * y;
        }
        Some(sum)
    }

    /// Compute cross product of two 3D vectors
    ///
    /// # Arguments
    /// * `a` - First 3D vector
    /// * `b` - Second 3D vector
    ///
    /// # Returns
    /// Cross product vector
    pub fn cross_product(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_norm_l2() {
        let array = Array3::from_shape_vec((2, 2, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap();
        let norm = VectorOperations::norm_l2(&array);

        // Expected: sqrt(1² + 2² + ... + 8²) = sqrt(204) ≈ 14.282856857
        let expected = (1..=8).map(|x| (x * x) as f64).sum::<f64>().sqrt();
        assert!((norm - expected).abs() < 1e-10);
    }

    #[test]
    fn test_vector_normalization() {
        let mut vector = vec![3.0, 4.0];
        let success = VectorOperations::normalize_vector(&mut vector);

        assert!(success);
        assert!((VectorOperations::vector_norm_l2(&vector) - 1.0).abs() < 1e-10);
        assert!((vector[0] - 0.6).abs() < 1e-6);
        assert!((vector[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_zero_vector_normalization() {
        let mut vector = vec![0.0, 0.0];
        let success = VectorOperations::normalize_vector(&mut vector);

        assert!(!success);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let dot = VectorOperations::dot_product(&a, &b).unwrap();
        assert!((dot - 32.0).abs() < 1e-10); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_dot_product_mismatched_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];

        assert!(VectorOperations::dot_product(&a, &b).is_none());
    }

    #[test]
    fn test_cross_product() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];

        let cross = VectorOperations::cross_product(&a, &b);
        assert!((cross[0] - 0.0).abs() < 1e-10);
        assert!((cross[1] - 0.0).abs() < 1e-10);
        assert!((cross[2] - 1.0).abs() < 1e-10);
    }
}
