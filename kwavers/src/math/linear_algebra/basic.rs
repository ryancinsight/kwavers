//! Basic Linear Algebra Operations
//!
//! This module provides fundamental linear algebra operations for real-valued matrices
//! and vectors, including system solving, matrix inversion, and basic decompositions.

use crate::core::error::{KwaversError, KwaversResult, NumericalError};
use nalgebra::DMatrix;
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
    /// Tuple (U, S, V) where U is (m×m) or (m×min(m,n)), S is diagonal (min(m,n)), V is (n×n) or (n×min(m,n))
    pub fn svd(matrix: &Array2<f64>) -> KwaversResult<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        let (m, n) = (matrix.nrows(), matrix.ncols());

        // Convert ndarray to nalgebra DMatrix
        let mut na_matrix = DMatrix::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                na_matrix[(i, j)] = matrix[[i, j]];
            }
        }

        // Compute SVD using nalgebra
        // We request both U and V^T
        let svd = na_matrix.svd(true, true);

        // Extract U
        let u_na = svd.u.ok_or_else(|| {
            KwaversError::Numerical(NumericalError::SolverFailed {
                method: "SVD".to_string(),
                reason: "Failed to compute left singular vectors (U)".to_string(),
            })
        })?;

        // Extract S
        let s_na = svd.singular_values;

        // Extract V^T (nalgebra returns V^T in v_t)
        let vt_na = svd.v_t.ok_or_else(|| {
            KwaversError::Numerical(NumericalError::SolverFailed {
                method: "SVD".to_string(),
                reason: "Failed to compute right singular vectors (V^T)".to_string(),
            })
        })?;

        // Convert U to ndarray
        let (u_rows, u_cols) = u_na.shape();
        let mut u = Array2::zeros((u_rows, u_cols));
        for i in 0..u_rows {
            for j in 0..u_cols {
                u[[i, j]] = u_na[(i, j)];
            }
        }

        // Convert S to ndarray
        let s_len = s_na.len();
        let mut s = Array1::zeros(s_len);
        for i in 0..s_len {
            s[i] = s_na[i];
        }

        // Convert V^T to ndarray
        let (vt_rows, vt_cols) = vt_na.shape();
        let mut vt = Array2::zeros((vt_rows, vt_cols));
        for i in 0..vt_rows {
            for j in 0..vt_cols {
                vt[[i, j]] = vt_na[(i, j)];
            }
        }

        // Return (U, S, V). V = vt^T
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

    #[test]
    fn test_svd_reconstruction() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let (u, s, v) = BasicLinearAlgebra::svd(&a).unwrap();

        // Check A = U * S * V^T
        // Construct S as a diagonal matrix
        let mut s_mat = Array2::zeros((2, 2));
        for i in 0..2 {
            s_mat[[i, i]] = s[i];
        }

        let reconstructed = u.dot(&s_mat).dot(&v.t());

        // Check reconstruction error
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (reconstructed[[i, j]] - a[[i, j]]).abs() < 1e-10,
                    "Reconstruction mismatch at [{}, {}]: expected {}, got {}",
                    i,
                    j,
                    a[[i, j]],
                    reconstructed[[i, j]]
                );
            }
        }

        // Check U is orthogonal: U^T * U = I
        let u_ortho = u.t().dot(&u);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (u_ortho[[i, j]] - expected).abs() < 1e-10,
                    "U not orthogonal"
                );
            }
        }

        // Check V is orthogonal: V^T * V = I
        let v_ortho = v.t().dot(&v);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (v_ortho[[i, j]] - expected).abs() < 1e-10,
                    "V not orthogonal"
                );
            }
        }
    }
}
