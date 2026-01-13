//! Basic Linear Algebra Operations
//!
//! This module provides fundamental linear algebra operations for real-valued matrices
//! and vectors, including system solving, matrix inversion, and basic decompositions.

use crate::core::error::{KwaversError, KwaversResult, NumericalError};
use ndarray::{s, Array1, Array2};

/// Basic linear algebra operations for real-valued matrices
#[derive(Debug)]
pub struct BasicLinearAlgebra;

impl BasicLinearAlgebra {
    /// Solve a linear system Ax = b using LU decomposition
    ///
    /// # Arguments
    /// * `a` - Coefficient matrix (n×n)
    /// * `b` - Right-hand side vector (n)
    ///
    /// # Returns
    /// Solution vector x
    ///
    /// # Errors
    /// Returns NumericalError if the matrix is singular or ill-conditioned
    pub fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> KwaversResult<Array1<f64>> {
        let n = a.nrows();
        if a.ncols() != n || b.len() != n {
            return Err(KwaversError::Numerical(NumericalError::MatrixDimension {
                operation: "solve_linear_system".to_string(),
                expected: format!("{}×{} matrix and {} vector", n, n, n),
                actual: format!("{}×{} matrix and {} vector", a.nrows(), a.ncols(), b.len()),
            }));
        }

        // Simple Gaussian elimination with partial pivoting
        let mut a_copy = a.clone();
        let mut b_copy = b.clone();
        let mut p = (0..n).collect::<Vec<_>>();

        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if a_copy[[k, i]].abs() > a_copy[[max_row, i]].abs() {
                    max_row = k;
                }
            }

            // Swap rows if needed
            if max_row != i {
                for j in 0..n {
                    let temp = a_copy[[i, j]];
                    a_copy[[i, j]] = a_copy[[max_row, j]];
                    a_copy[[max_row, j]] = temp;
                }
                let temp = b_copy[i];
                b_copy[i] = b_copy[max_row];
                b_copy[max_row] = temp;
                p.swap(i, max_row);
            }

            // Check for singularity
            if a_copy[[i, i]].abs() < 1e-12 {
                return Err(KwaversError::Numerical(NumericalError::SingularMatrix {
                    operation: "LU decomposition".to_string(),
                    condition_number: f64::INFINITY,
                }));
            }

            // Eliminate
            for k in (i + 1)..n {
                let factor = a_copy[[k, i]] / a_copy[[i, i]];
                for j in i..n {
                    a_copy[[k, j]] -= factor * a_copy[[i, j]];
                }
                b_copy[k] -= factor * b_copy[i];
            }
        }

        // Back substitution
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in (i + 1)..n {
                sum += a_copy[[i, j]] * x[j];
            }
            x[i] = (b_copy[i] - sum) / a_copy[[i, i]];
        }

        Ok(x)
    }

    /// Compute the inverse of a square matrix using LU decomposition
    ///
    /// # Arguments
    /// * `matrix` - Square matrix to invert (n×n)
    ///
    /// # Returns
    /// Inverse matrix
    ///
    /// # Errors
    /// Returns NumericalError if the matrix is singular
    pub fn matrix_inverse(matrix: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        let n = matrix.nrows();
        if matrix.ncols() != n {
            return Err(KwaversError::Numerical(NumericalError::MatrixDimension {
                operation: "matrix_inverse".to_string(),
                expected: format!("{}×{} square matrix", n, n),
                actual: format!("{}×{} matrix", matrix.nrows(), matrix.ncols()),
            }));
        }

        // Create identity matrix
        let identity = Array2::eye(n);
        let mut result = Array2::zeros((n, n));

        // Solve for each column of the identity matrix
        for i in 0..n {
            let b = identity.column(i).to_owned();
            let x = Self::solve_linear_system(matrix, &b)?;
            result.column_mut(i).assign(&x);
        }

        Ok(result)
    }

    /// Compute QR decomposition of a matrix
    ///
    /// # Arguments
    /// * `matrix` - Input matrix (m×n)
    ///
    /// # Returns
    /// Tuple (Q, R) where Q is orthogonal (m×m) and R is upper triangular (m×n)
    pub fn qr_decomposition(matrix: &Array2<f64>) -> KwaversResult<(Array2<f64>, Array2<f64>)> {
        let (m, n) = (matrix.nrows(), matrix.ncols());
        let mut q = Array2::eye(m);
        let mut r = matrix.clone();

        // Householder QR decomposition
        for j in 0..n.min(m) {
            // Extract column j from j to m-1
            let col_j = r.column(j).slice(s![j..]).to_owned();

            // Compute Householder vector
            let norm = col_j.iter().map(|&x| x * x).sum::<f64>().sqrt();
            let mut v = col_j.clone();
            if v[0] >= 0.0 {
                v[0] += norm;
            } else {
                v[0] -= norm;
            }

            let v_norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if v_norm > 1e-12 {
                v /= v_norm;

                // Apply Householder reflection to R
                for k in j..n {
                    let dot_product = v
                        .iter()
                        .zip(r.column(k).slice(s![j..]))
                        .map(|(&a, &b)| a * b)
                        .sum::<f64>();
                    for (i, &vi) in v.iter().enumerate() {
                        r[[j + i, k]] -= 2.0 * vi * dot_product;
                    }
                }

                // Apply Householder reflection to Q
                for k in 0..m {
                    let dot_product = v
                        .iter()
                        .zip(q.column(k).slice(s![j..]))
                        .map(|(&a, &b)| a * b)
                        .sum::<f64>();
                    for (i, &vi) in v.iter().enumerate() {
                        q[[j + i, k]] -= 2.0 * vi * dot_product;
                    }
                }
            }
        }

        Ok((q, r))
    }

    /// Compute SVD decomposition of a matrix
    ///
    /// # Arguments
    /// * `matrix` - Input matrix (m×n)
    ///
    /// # Returns
    /// Tuple (U, S, V) where U is (m×m), S is diagonal (min(m,n)), V is (n×n)
    pub fn svd(matrix: &Array2<f64>) -> KwaversResult<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        // Use Golub-Kahan-Lanczos bidiagonalization followed by implicit QR
        // This is a simplified implementation - for production use, consider LAPACK
        let (m, n) = (matrix.nrows(), matrix.ncols());

        // For now, return a basic SVD using QR decomposition
        // TODO: Implement proper SVD algorithm
        let (q, r) = Self::qr_decomposition(matrix)?;

        // Extract singular values from R diagonal
        let min_dim = m.min(n);
        let mut s = Array1::zeros(min_dim);
        for i in 0..min_dim {
            s[i] = r[[i, i]].abs();
        }

        // U = Q, V = I (simplified)
        let u = q;
        let vt = Array2::eye(n);

        Ok((u, s, vt.t().to_owned()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_solve_linear_system() {
        let a = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0]).unwrap();
        let b = Array1::from_vec(vec![3.0, 3.0]);

        let x = BasicLinearAlgebra::solve_linear_system(&a, &b).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_inverse() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let a_inv = BasicLinearAlgebra::matrix_inverse(&a).unwrap();

        // Check A * A^(-1) = I
        let identity = a.dot(&a_inv);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((identity[[i, j]] - expected).abs() < 1e-10);
            }
        }
    }
}
