//! Linear Algebra Operations
//!
//! This module provides comprehensive linear algebra operations organized into
//! focused submodules for better maintainability and clear separation of concerns.
//!
//! ## Submodules
//!
//! - `basic`: Fundamental operations for real matrices (solving, inversion, decompositions)
//! - `complex`: Complex matrix operations for beamforming applications
//! - `eigen`: Eigenvalue decomposition for subspace methods
//! - `norms`: Vector norms and basic vector operations
//! - `sparse`: Sparse matrix operations
//!
//! ## Design Principles
//!
//! - **Single Responsibility**: Each submodule handles one category of operations
//! - **Zero Dependencies**: Pure Rust implementations without external BLAS/LAPACK
//! - **Type Safety**: Strong typing with compile-time dimension checking where possible
//! - **Performance**: SIMD-friendly algorithms optimized for cache locality

pub mod basic;
pub mod complex;
pub mod eigen;
pub mod eigendecomposition;
pub mod iterative;
pub mod norms;
pub mod sparse;

// Re-export types for backward compatibility
pub use basic::BasicLinearAlgebra;
pub use complex::ComplexLinearAlgebra;
pub use eigen::EigenDecomposition;
pub use eigendecomposition::{EigenResult, EigenSolver, EigenSolverConfig};
pub use norms::VectorOperations;

// Backward compatibility alias
pub use BasicLinearAlgebra as LinearAlgebra;

impl LinearAlgebra {
    /// Solve complex linear system Ax = b (backward compatibility)
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn solve_linear_system_complex(
        a: &Array2<Complex<f64>>,
        b: &Array1<Complex<f64>>,
    ) -> KwaversResult<Array1<Complex<f64>>> {
        ComplexLinearAlgebra::solve_linear_system_complex(a, b)
    }

    /// Compute inverse of a complex matrix (backward compatibility)
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn matrix_inverse_complex(
        matrix: &Array2<Complex<f64>>,
    ) -> KwaversResult<Array2<Complex<f64>>> {
        ComplexLinearAlgebra::matrix_inverse_complex(matrix)
    }

    /// Compute eigendecomposition of symmetric matrix (backward compatibility)
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn eigendecomposition(matrix: &Array2<f64>) -> KwaversResult<(Array1<f64>, Array2<f64>)> {
        EigenDecomposition::eigendecomposition(matrix)
    }

    /// Compute eigendecomposition of Hermitian matrix (backward compatibility)
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn hermitian_eigendecomposition_complex(
        matrix: &Array2<Complex<f64>>,
    ) -> KwaversResult<(Array1<f64>, Array2<Complex<f64>>)> {
        EigenDecomposition::hermitian_eigendecomposition_complex(matrix)
    }
}

use crate::core::error::KwaversResult;
use ndarray::{Array1, Array2};
use num_complex::Complex;
use num_traits::{Float, NumCast, Zero};

/// Generic numeric operations for improved type safety and reusability
pub trait NumericOps<T>: Clone + Copy + PartialOrd + Zero
where
    T: Float + NumCast,
{
    /// Generic dot product for any float type
    fn dot_product(a: &[T], b: &[T]) -> Option<T> {
        if a.len() != b.len() {
            return None;
        }
        Some(
            a.iter()
                .zip(b.iter())
                .map(|(&x, &y)| x * y)
                .fold(T::zero(), |acc, val| acc + val),
        )
    }

    /// Generic vector normalization
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn normalize(vector: &mut [T]) -> bool {
        let norm_sq = vector
            .iter()
            .map(|&x| x * x)
            .fold(T::zero(), |acc, val| acc + val);
        if norm_sq <= T::zero() {
            return false;
        }
        let norm = norm_sq.sqrt();
        for x in vector.iter_mut() {
            *x = *x / norm;
        }
        true
    }

    /// Generic element-wise addition for arrays
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn add_arrays(a: &[T], b: &[T], out: &mut [T]) -> Result<(), &'static str> {
        if a.len() != b.len() || b.len() != out.len() {
            return Err("Array length mismatch");
        }
        for ((a_val, b_val), out_val) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
            *out_val = *a_val + *b_val;
        }
        Ok(())
    }

    /// Generic scalar multiplication
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn scale_array(input: &[T], scalar: T, out: &mut [T]) -> Result<(), &'static str> {
        if input.len() != out.len() {
            return Err("Array length mismatch");
        }
        for (input_val, out_val) in input.iter().zip(out.iter_mut()) {
            *out_val = *input_val * scalar;
        }
        Ok(())
    }

    /// Generic L2 norm calculation
    fn l2_norm(array: &[T]) -> T {
        array
            .iter()
            .map(|&x| x * x)
            .fold(T::zero(), |acc, val| acc + val)
            .sqrt()
    }

    /// Generic maximum absolute value
    fn max_abs(array: &[T]) -> T {
        array
            .iter()
            .map(|&x| x.abs())
            .fold(T::zero(), |acc, val| acc.max(val))
    }

    /// Safe division with tolerance check
    fn safe_divide(numerator: T, denominator: T, tolerance: T) -> Option<T> {
        if denominator.abs() > tolerance {
            Some(numerator / denominator)
        } else {
            None
        }
    }
}

/// Tolerance constants for numerical operations
pub mod tolerance {
    /// Default tolerance for convergence checks
    pub const DEFAULT: f64 = 1e-12;
    /// Tolerance for matrix rank determination
    pub const RANK: f64 = 1e-10;
    /// Maximum iterations for iterative methods
    pub const MAX_ITERATIONS: usize = 1000;

    /// Tolerance used for detecting near-zero complex pivots during LU factorization.
    ///
    /// This is intentionally aligned with `RANK` to preserve existing conditioning policy.
    pub const COMPLEX_PIVOT: f64 = RANK;

    /// Convergence tolerance for the SSOT complex Hermitian eigensolver (Jacobi on real-embedded form).
    ///
    /// This bounds the maximum absolute off-diagonal entry of the embedded real symmetric matrix
    /// before declaring convergence.
    pub const HERMITIAN_EIG_TOL: f64 = 1e-12;

    /// Maximum sweeps (major iterations) for the SSOT complex Hermitian eigensolver (Jacobi).
    pub const HERMITIAN_EIG_MAX_SWEEPS: usize = 2048;

    /// Convergence tolerance for tridiagonal QR eigensolver (off-diagonal magnitude threshold).
    pub const SYMM_TRIDIAG_QR_TOL: f64 = 1e-12;

    /// Maximum iterations for implicit QR on symmetric tridiagonal matrices.
    pub const SYMM_TRIDIAG_QR_MAX_ITERS: usize = 256;

    /// Below this dimension (2n for the embedded real symmetric problem), Jacobi is fine and
    /// often faster due to lower constant factors and simpler code paths.
    pub const HERMITIAN_EIG_JACOBI_CUTOFF_DIM: usize = 64;
}

// Implement NumericOps for standard float types
impl NumericOps<Self> for f64 {}
impl NumericOps<Self> for f32 {}

/// Compute L2 norm of a 3D array (convenience function)
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
#[must_use]
pub fn norm_l2(array: &ndarray::Array3<f64>) -> f64 {
    VectorOperations::norm_l2(array)
}

/// Extension trait for ndarray operations
pub trait LinearAlgebraExt<T> {
    /// Solve linear system in-place where possible
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn solve_into(&self, b: Array1<T>) -> KwaversResult<Array1<T>>;

    /// Compute matrix inverse
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn inv(&self) -> KwaversResult<Array2<T>>;

    /// Eigendecomposition
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn eig(&self) -> KwaversResult<(Array1<T>, Array2<T>)>;
}

impl LinearAlgebraExt<f64> for Array2<f64> {
    fn solve_into(&self, b: Array1<f64>) -> KwaversResult<Array1<f64>> {
        LinearAlgebra::solve_linear_system(self, &b)
    }

    fn inv(&self) -> KwaversResult<Self> {
        LinearAlgebra::matrix_inverse(self)
    }

    fn eig(&self) -> KwaversResult<(Array1<f64>, Self)> {
        LinearAlgebra::eigendecomposition(self)
    }
}

impl LinearAlgebraExt<Complex<f64>> for Array2<Complex<f64>> {
    fn solve_into(&self, b: Array1<Complex<f64>>) -> KwaversResult<Array1<Complex<f64>>> {
        LinearAlgebra::solve_linear_system_complex(self, &b)
    }

    fn inv(&self) -> KwaversResult<Self> {
        LinearAlgebra::matrix_inverse_complex(self)
    }

    fn eig(&self) -> KwaversResult<(Array1<Complex<f64>>, Self)> {
        let (eigenvalues, eigenvectors) =
            LinearAlgebra::hermitian_eigendecomposition_complex(self)?;
        Ok((
            eigenvalues.mapv(|lambda| Complex::new(lambda, 0.0)),
            eigenvectors,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_linear_algebra_re_exports() {
        // Test that re-exports work
        let a = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0]).unwrap();
        let b = Array1::from_vec(vec![3.0, 3.0]);

        let x = LinearAlgebra::solve_linear_system(&a, &b).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_norm_l2_convenience_function() {
        let array = ndarray::Array3::from_shape_vec(
            (2, 2, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();
        let norm = norm_l2(&array);

        // Expected: sqrt(1² + 2² + ... + 8²) = sqrt(204) ≈ 14.282856857
        let expected = (1..=8).map(|x| (x * x) as f64).sum::<f64>().sqrt();
        assert!((norm - expected).abs() < 1e-10);
    }

    #[test]
    fn test_linear_algebra_ext_trait() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Array1::from_vec(vec![5.0, 11.0]);

        // Test solve_into method
        let x = a.solve_into(b).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-6);
        assert!((x[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn complex_ext_eig_delegates_to_hermitian_solver() {
        let matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex::new(2.0, 0.0),
                Complex::new(1.0, -1.0),
                Complex::new(1.0, 1.0),
                Complex::new(3.0, 0.0),
            ],
        )
        .unwrap();

        let (eigenvalues, eigenvectors) = matrix.eig().unwrap();

        assert!((eigenvalues[0] - Complex::new(4.0, 0.0)).norm() < 1e-10);
        assert!((eigenvalues[1] - Complex::new(1.0, 0.0)).norm() < 1e-10);

        for column in 0..2 {
            let lambda = eigenvalues[column];
            let vector = eigenvectors.column(column).to_owned();
            let residual = matrix.dot(&vector) - vector.mapv(|entry| lambda * entry);
            assert!(residual.iter().all(|entry| entry.norm() < 1e-10));
        }
    }

    #[test]
    fn complex_ext_eig_rejects_non_hermitian_matrix() {
        let matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(1.0, 1.0),
                Complex::new(2.0, 1.0),
                Complex::new(3.0, 0.0),
            ],
        )
        .unwrap();

        let error = matrix.eig().unwrap_err();

        assert!(format!("{error}").contains("not Hermitian"));
    }
}
