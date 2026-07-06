//! Basic Linear Algebra Operations
//!
//! This module provides fundamental linear algebra operations for real-valued matrices
//! and vectors, including system solving, matrix inversion, and basic decompositions.

use kwavers_core::error::{KwaversError, KwaversResult, NumericalError};
use leto::Array2 as LetoArray2;
use leto_ops::{qr_decompose, svd_rank_revealing};
use ndarray::{Array1, Array2};
use std::fmt::Display;

/// Basic linear algebra operations for real-valued matrices
#[derive(Debug)]
pub struct LinearAlgebra;

impl LinearAlgebra {
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
                operation: "solve_linear_system".to_owned(),
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
                    operation: "LU decomposition".to_owned(),
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
                operation: "matrix_inverse".to_owned(),
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

    /// Compute the Householder QR decomposition of a matrix.
    ///
    /// # Arguments
    /// * `matrix` - Input matrix (m×n)
    ///
    /// # Returns
    /// Tuple `(Q, R)` with `Q` (m×m) orthogonal and `R` (m×n) upper
    /// triangular such that `A = Q·R`. Delegates to Leto's native Householder QR.
    /// # Errors
    /// - Returns [`Err`] if `matrix` is underdetermined (`m < n`), contains a
    ///   non-finite value, or is rank-deficient under Leto's QR contract.
    ///
    pub fn qr_decomposition(matrix: &Array2<f64>) -> KwaversResult<(Array2<f64>, Array2<f64>)> {
        let leto_matrix = ndarray_to_leto(matrix);
        let qr =
            qr_decompose(&leto_matrix.view()).map_err(leto_linalg_error("QR decomposition"))?;
        let q = leto_to_ndarray(qr.q(), "QR Q")?;
        let r = leto_to_ndarray(qr.r(), "QR R")?;
        Ok((q, r))
    }

    /// Compute SVD decomposition of a matrix
    ///
    /// # Arguments
    /// * `matrix` - Input matrix (m×n)
    ///
    /// # Returns
    /// Tuple `(U, S, V)` where `U` is `(m×min(m,n))`, `S` has length
    /// `min(m,n)`, and `V` is `(n×min(m,n))`.
    /// # Errors
    /// - Returns [`Err`] if Leto rejects the input shape or values.
    ///
    pub fn svd(matrix: &Array2<f64>) -> KwaversResult<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        let leto_matrix = ndarray_to_leto(matrix);
        let svd = svd_rank_revealing(&leto_matrix.view()).map_err(leto_linalg_error("SVD"))?;
        let u = leto_to_ndarray(svd.left_singular_vectors, "SVD U")?;
        let s = Array1::from_vec(svd.singular_values);
        let v = leto_to_ndarray(svd.right_singular_vectors, "SVD V")?;

        Ok((u, s, v))
    }
}

fn ndarray_to_leto(matrix: &Array2<f64>) -> LetoArray2<f64> {
    matrix.clone().into()
}

fn leto_to_ndarray(matrix: LetoArray2<f64>, method: &'static str) -> KwaversResult<Array2<f64>> {
    matrix.try_into().map_err(|error| {
        KwaversError::Numerical(NumericalError::SolverFailed {
            method: method.to_owned(),
            reason: format!("failed to convert Leto matrix result to ndarray: {error}"),
        })
    })
}

fn leto_linalg_error(
    method: &'static str,
) -> impl FnOnce(leto::LetoError) -> KwaversError + 'static {
    move |error| {
        KwaversError::Numerical(NumericalError::SolverFailed {
            method: method.to_owned(),
            reason: leto_error_reason(error),
        })
    }
}

fn leto_error_reason(error: impl Display) -> String {
    error.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_solve_linear_system() {
        let a = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0]).unwrap();
        let b = Array1::from_vec(vec![3.0, 3.0]);

        let x = LinearAlgebra::solve_linear_system(&a, &b).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_inverse() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let a_inv = LinearAlgebra::matrix_inverse(&a).unwrap();

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
    fn test_qr_reconstruction_and_orthogonality() {
        // Square and over-determined (m > n) cases: A = Q·R with Qᵀ Q = I.
        let cases = [
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 0.0, 0.0, 1.0, 3.0, 4.0, 0.0, 1.0])
                .unwrap(),
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap(),
        ];
        for a in &cases {
            let (q, r) = LinearAlgebra::qr_decomposition(a).unwrap();
            // Reconstruction A = Q·R.
            let recon = q.dot(&r);
            assert_eq!(recon.dim(), a.dim());
            for i in 0..a.nrows() {
                for j in 0..a.ncols() {
                    assert!(
                        (recon[[i, j]] - a[[i, j]]).abs() < 1e-9,
                        "QR reconstruction mismatch at [{i},{j}]: {} vs {}",
                        recon[[i, j]],
                        a[[i, j]]
                    );
                }
            }
            // Q has orthonormal columns: Qᵀ Q = I (k×k).
            let qtq = q.t().dot(&q);
            for i in 0..qtq.nrows() {
                for j in 0..qtq.ncols() {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!((qtq[[i, j]] - expected).abs() < 1e-9, "Q not orthonormal");
                }
            }
            // R is upper triangular.
            for i in 0..r.nrows() {
                for j in 0..i.min(r.ncols()) {
                    assert!(
                        r[[i, j]].abs() < 1e-9,
                        "R not upper triangular at [{i},{j}]"
                    );
                }
            }
        }
    }

    #[test]
    fn test_svd_reconstruction() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let (u, s, v) = LinearAlgebra::svd(&a).unwrap();

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
