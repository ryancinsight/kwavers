//! Pure Rust Linear Algebra Operations
//!
//! This module provides essential linear algebra operations without external dependencies,
//! following SOLID principles and zero-copy techniques where possible.

use crate::error::{KwaversError, KwaversResult, NumericalError};
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
        Some(a.iter().zip(b.iter()).map(|(&x, &y)| x * y).fold(T::zero(), |acc, val| acc + val))
    }
    
    /// Generic vector normalization
    fn normalize(vector: &mut [T]) -> bool {
        let norm_sq = vector.iter().map(|&x| x * x).fold(T::zero(), |acc, val| acc + val);
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
        array.iter().map(|&x| x * x).fold(T::zero(), |acc, val| acc + val).sqrt()
    }

    /// Generic maximum absolute value
    fn max_abs(array: &[T]) -> T {
        array.iter().map(|&x| x.abs()).fold(T::zero(), |acc, val| acc.max(val))
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
}

// Implement NumericOps for standard float types
impl NumericOps<f64> for f64 {}
impl NumericOps<f32> for f32 {}

/// Pure Rust implementation of basic linear algebra operations
#[derive(Debug)]
pub struct LinearAlgebra;

impl LinearAlgebra {
    /// Solve linear system Ax = b using LU decomposition with partial pivoting
    pub fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> KwaversResult<Array1<f64>> {
        let n = a.nrows();
        if a.ncols() != n || b.len() != n {
            return Err(KwaversError::Numerical(NumericalError::MatrixDimension {
                operation: "solve_linear_system".to_string(),
                expected: format!("{n}x{n}"),
                actual: format!("{}x{}", a.nrows(), a.ncols()),
            }));
        }

        let mut lu = a.clone();
        let mut perm = (0..n).collect::<Vec<_>>();

        // LU decomposition with partial pivoting
        for k in 0..n - 1 {
            // Find pivot
            let mut max_row = k;
            for i in k + 1..n {
                if lu[[i, k]].abs() > lu[[max_row, k]].abs() {
                    max_row = i;
                }
            }

            // Swap rows if needed
            if max_row != k {
                perm.swap(k, max_row);
                for j in 0..n {
                    let temp = lu[[k, j]];
                    lu[[k, j]] = lu[[max_row, j]];
                    lu[[max_row, j]] = temp;
                }
            }

            // Check for singular matrix
            if lu[[k, k]].abs() < tolerance::RANK {
                return Err(KwaversError::Numerical(NumericalError::SingularMatrix {
                    operation: "LU decomposition".to_string(),
                    condition_number: f64::INFINITY,
                }));
            }

            // Elimination
            for i in k + 1..n {
                lu[[i, k]] /= lu[[k, k]];
                for j in k + 1..n {
                    lu[[i, j]] -= lu[[i, k]] * lu[[k, j]];
                }
            }
        }

        // Apply permutation to b
        let mut pb = Array1::zeros(n);
        for i in 0..n {
            pb[i] = b[perm[i]];
        }

        // Forward substitution for Ly = Pb
        let mut y = Array1::zeros(n);
        for i in 0..n {
            y[i] = pb[i];
            for j in 0..i {
                y[i] -= lu[[i, j]] * y[j];
            }
        }

        // Back substitution for Ux = y
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            x[i] = y[i];
            for j in i + 1..n {
                x[i] -= lu[[i, j]] * x[j];
            }
            x[i] /= lu[[i, i]];
        }

        Ok(x)
    }

    /// Compute eigenvalues and eigenvectors using QR algorithm
    pub fn eigendecomposition(matrix: &Array2<f64>) -> KwaversResult<(Array1<f64>, Array2<f64>)> {
        let n = matrix.nrows();
        if matrix.ncols() != n {
            return Err(KwaversError::Numerical(NumericalError::MatrixDimension {
                operation: "eigendecomposition".to_string(),
                expected: format!("{n}x{n}"),
                actual: format!("{}x{}", matrix.nrows(), matrix.ncols()),
            }));
        }

        // Use power iteration for dominant eigenvalue/eigenvector
        // Implementation using QR algorithm for eigenvalue decomposition
        let mut a = matrix.clone();
        let mut q = Array2::eye(n);

        // QR iterations for eigenvalue computation
        for _ in 0..tolerance::MAX_ITERATIONS {
            let (q_iteration, r) = Self::qr_decomposition(&a)?;
            a = r.dot(&q_iteration);
            q = q.dot(&q_iteration);

            // Check for convergence using off-diagonal norm
            let mut converged = true;
            for i in 0..n - 1 {
                if a[[i + 1, i]].abs() > tolerance::DEFAULT {
                    converged = false;
                    break;
                }
            }

            if converged {
                break;
            }
        }

        // Extract eigenvalues from diagonal
        let eigenvalues = Array1::from_iter((0..n).map(|i| a[[i, i]]));

        Ok((eigenvalues, q))
    }

    /// QR decomposition using Gram-Schmidt process
    pub fn qr_decomposition(matrix: &Array2<f64>) -> KwaversResult<(Array2<f64>, Array2<f64>)> {
        let (m, n) = matrix.dim();
        let mut q = Array2::zeros((m, n));
        let mut r = Array2::zeros((n, n));

        for j in 0..n {
            let mut v = matrix.column(j).to_owned();

            // Orthogonalize against previous columns
            for i in 0..j {
                let q_i = q.column(i);
                r[[i, j]] = v.dot(&q_i);
                v = v - r[[i, j]] * &q_i;
            }

            // Normalize
            r[[j, j]] = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if r[[j, j]] < tolerance::RANK {
                return Err(KwaversError::Numerical(NumericalError::SingularMatrix {
                    operation: "QR decomposition".to_string(),
                    condition_number: f64::INFINITY,
                }));
            }

            for k in 0..m {
                q[[k, j]] = v[k] / r[[j, j]];
            }
        }

        Ok((q, r))
    }

    /// Matrix inversion using LU decomposition
    pub fn matrix_inverse(matrix: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        let n = matrix.nrows();
        if matrix.ncols() != n {
            return Err(KwaversError::Numerical(NumericalError::MatrixDimension {
                operation: "matrix_inverse".to_string(),
                expected: format!("{n}x{n}"),
                actual: format!("{}x{}", matrix.nrows(), matrix.ncols()),
            }));
        }

        let mut inverse = Array2::zeros((n, n));
        let identity = Array2::eye(n);

        // Solve Ax = ei for each column of identity matrix
        for i in 0..n {
            let column = identity.column(i);
            let solution = Self::solve_linear_system(matrix, &column.to_owned())?;
            for j in 0..n {
                inverse[[j, i]] = solution[j];
            }
        }

        Ok(inverse)
    }

    /// Singular Value Decomposition
    ///
    /// Note: This implementation uses the eigendecomposition of A^T*A which is numerically
    /// less stable than methods like Golub-Kahan bidiagonalization. For production use,
    /// consider using LAPACK bindings or a more robust algorithm.
    ///
    /// Reference: Golub & Van Loan (2013) "Matrix Computations", 4th ed., Section 8.6
    pub fn svd(matrix: &Array2<f64>) -> KwaversResult<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        let (m, n) = matrix.dim();

        // Compute A^T * A for V and singular values
        let ata = matrix.t().dot(matrix);
        let (s_squared, v) = Self::eigendecomposition(&ata)?;

        // Singular values are square roots of eigenvalues
        let s = Array1::from_iter(s_squared.iter().map(|&x| x.sqrt()));

        // Compute U = A * V * S^(-1)
        let mut u = Array2::zeros((m, n.min(m)));
        for i in 0..n.min(m) {
            if s[i] > tolerance::RANK {
                let v_col = v.column(i);
                let u_col = matrix.dot(&v_col) / s[i];
                for j in 0..m {
                    u[[j, i]] = u_col[j];
                }
            }
        }

        Ok((u, s, v))
    }

    /// Compute L2 norm of a 3D array
    pub fn norm_l2_3d(array: &ndarray::Array3<f64>) -> f64 {
        array.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }
}

/// Compute L2 norm of a 3D array (convenience function)
pub fn norm_l2(array: &ndarray::Array3<f64>) -> f64 {
    LinearAlgebra::norm_l2_3d(array)
}

/// Extension trait for ndarray operations
pub trait LinearAlgebraExt<T> {
    /// Solve linear system in-place where possible
    fn solve_into(&self, b: Array1<T>) -> KwaversResult<Array1<T>>;

    /// Compute matrix inverse
    fn inv(&self) -> KwaversResult<Array2<T>>;

    /// Eigendecomposition
    fn eig(&self) -> KwaversResult<(Array1<T>, Array2<T>)>;
}

impl LinearAlgebraExt<f64> for Array2<f64> {
    fn solve_into(&self, b: Array1<f64>) -> KwaversResult<Array1<f64>> {
        LinearAlgebra::solve_linear_system(self, &b)
    }

    fn inv(&self) -> KwaversResult<Array2<f64>> {
        LinearAlgebra::matrix_inverse(self)
    }

    fn eig(&self) -> KwaversResult<(Array1<f64>, Array2<f64>)> {
        LinearAlgebra::eigendecomposition(self)
    }
}

impl LinearAlgebraExt<Complex<f64>> for Array2<Complex<f64>> {
    fn solve_into(&self, _b: Array1<Complex<f64>>) -> KwaversResult<Array1<Complex<f64>>> {
        // Complex number support would require separate implementation
        Err(KwaversError::Numerical(
            NumericalError::UnsupportedOperation {
                operation: "Complex linear solve".to_string(),
                reason: "Complex number support not implemented".to_string(),
            },
        ))
    }

    fn inv(&self) -> KwaversResult<Array2<Complex<f64>>> {
        // Complex number support would require separate implementation
        Err(KwaversError::Numerical(
            NumericalError::UnsupportedOperation {
                operation: "Complex matrix inverse".to_string(),
                reason: "Complex number support not implemented".to_string(),
            },
        ))
    }

    fn eig(&self) -> KwaversResult<(Array1<Complex<f64>>, Array2<Complex<f64>>)> {
        // Complex number support would require separate implementation
        Err(KwaversError::Numerical(
            NumericalError::UnsupportedOperation {
                operation: "Complex eigendecomposition".to_string(),
                reason: "Complex number support not implemented".to_string(),
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_relative_eq, assert_abs_diff_eq};

    #[test]
    fn test_numeric_ops_add_arrays() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let mut out = [0.0; 3];

        assert!(f64::add_arrays(&a, &b, &mut out).is_ok());
        assert_eq!(out, [5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_numeric_ops_scale_array() {
        let input = [1.0, 2.0, 3.0];
        let mut out = [0.0; 3];

        assert!(f64::scale_array(&input, 2.0, &mut out).is_ok());
        assert_eq!(out, [2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_numeric_ops_l2_norm() {
        let array = [3.0, 4.0];
        let norm = f64::l2_norm(&array);
        assert_relative_eq!(norm, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_numeric_ops_safe_divide() {
        assert_eq!(f64::safe_divide(10.0, 2.0, 1e-10), Some(5.0));
        assert_eq!(f64::safe_divide(10.0, 1e-12, 1e-10), None);
    }

    #[test]
    fn test_numeric_ops_max_abs() {
        let array = [-5.0, 3.0, -2.0, 4.0];
        assert_eq!(f64::max_abs(&array), 5.0);
    }

    #[test]
    fn test_generic_float_types() {
        // Test that the trait works with both f64 and f32
        let a_f64 = [1.0_f64, 2.0, 3.0];
        let norm_f64 = f64::l2_norm(&a_f64);
        
        let a_f32 = [1.0_f32, 2.0, 3.0];
        let norm_f32 = f32::l2_norm(&a_f32);
        
        assert_relative_eq!(norm_f64 as f32, norm_f32, epsilon = 1e-6);
    }

    #[test]
    fn test_solve_linear_system() {
        let a = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 1.0]).unwrap();
        let b = Array1::from_vec(vec![3.0, 2.0]);

        let x = LinearAlgebra::solve_linear_system(&a, &b).unwrap();

        // Solution should be [1, 1]
        assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_inverse() {
        let a = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 1.0]).unwrap();
        let inv = LinearAlgebra::matrix_inverse(&a).unwrap();

        // Check A * A^(-1) = I
        let product = a.dot(&inv);
        assert_abs_diff_eq!(product[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(product[[1, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(product[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(product[[1, 0]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_qr_decomposition() {
        let a = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 1.0, 2.0, 2.0, 3.0]).unwrap();
        let (q, r) = LinearAlgebra::qr_decomposition(&a).unwrap();

        // Check that Q * R = A
        let product = q.dot(&r);
        for i in 0..3 {
            for j in 0..2 {
                assert_abs_diff_eq!(product[[i, j]], a[[i, j]], epsilon = 1e-10);
            }
        }
    }
}
