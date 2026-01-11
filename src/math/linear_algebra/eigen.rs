//! Eigenvalue Decomposition Operations
//!
//! This module provides eigenvalue and eigenvector computation for both real
//! and complex Hermitian matrices, essential for subspace beamforming methods.

use crate::core::error::{KwaversError, KwaversResult, NumericalError};
use ndarray::{Array1, Array2};
use num_complex::Complex;

/// Eigenvalue decomposition operations
pub struct EigenDecomposition;

impl EigenDecomposition {
    /// Compute eigendecomposition of a real symmetric matrix
    ///
    /// # Arguments
    /// * `matrix` - Real symmetric matrix (n×n)
    ///
    /// # Returns
    /// Tuple (eigenvalues, eigenvectors) where eigenvalues is (n) and eigenvectors is (n×n)
    ///
    /// # Note
    /// Uses Jacobi eigenvalue algorithm - suitable for small matrices (< 100×100)
    pub fn eigendecomposition(matrix: &Array2<f64>) -> KwaversResult<(Array1<f64>, Array2<f64>)> {
        let n = matrix.nrows();
        if matrix.ncols() != n {
            return Err(KwaversError::Numerical(NumericalError::MatrixDimension {
                operation: "eigendecomposition".to_string(),
                expected: format!("{}×{} square matrix", n, n),
                actual: format!("{}×{} matrix", matrix.nrows(), matrix.ncols()),
            }));
        }

        // Check if matrix is symmetric
        for i in 0..n {
            for j in (i + 1)..n {
                if (matrix[[i, j]] - matrix[[j, i]]).abs() > 1e-10 {
                return Err(KwaversError::Numerical(NumericalError::InvalidOperation(
                    "Matrix must be symmetric for real eigendecomposition".to_string(),
                )));
                }
            }
        }

        let mut a = matrix.clone();
        let mut eigenvectors = Array2::eye(n);
        let mut eigenvalues = Array1::zeros(n);

        // Jacobi eigenvalue algorithm
        let mut max_iterations = 100;
        let tolerance = 1e-10;

        for _ in 0..max_iterations {
            // Find largest off-diagonal element
            let mut max_val = 0.0;
            let mut p = 0;
            let mut q = 1;

            for i in 0..n {
                for j in (i + 1)..n {
                    if a[[i, j]].abs() > max_val {
                        max_val = a[[i, j]].abs();
                        p = i;
                        q = j;
                    }
                }
            }

            // Check convergence
            if max_val < tolerance {
                break;
            }

            // Compute rotation angle
            let theta = if a[[p, p]] == a[[q, q]] {
                std::f64::consts::PI / 4.0
            } else {
                0.5 * (a[[q, q]] - a[[p, p]]) / a[[p, q]].atan2((a[[q, q]] - a[[p, p]]) / (2.0 * a[[p, q]]))
            };

            let c = theta.cos();
            let s = theta.sin();

            // Apply rotation to matrix A
            for i in 0..n {
                let a_ip = a[[i, p]];
                let a_iq = a[[i, q]];
                a[[i, p]] = c * a_ip - s * a_iq;
                a[[i, q]] = s * a_ip + c * a_iq;
            }

            for i in 0..n {
                let a_pi = a[[p, i]];
                let a_qi = a[[q, i]];
                a[[p, i]] = c * a_pi - s * a_qi;
                a[[q, i]] = s * a_pi + c * a_qi;
            }

            // Apply rotation to eigenvectors
            for i in 0..n {
                let v_ip = eigenvectors[[i, p]];
                let v_iq = eigenvectors[[i, q]];
                eigenvectors[[i, p]] = c * v_ip - s * v_iq;
                eigenvectors[[i, q]] = s * v_ip + c * v_iq;
            }
        }

        // Extract eigenvalues from diagonal
        for i in 0..n {
            eigenvalues[i] = a[[i, i]];
        }

        Ok((eigenvalues, eigenvectors))
    }

    /// Compute eigendecomposition of a complex Hermitian matrix
    ///
    /// # Arguments
    /// * `matrix` - Complex Hermitian matrix (n×n)
    ///
    /// # Returns
    /// Tuple (eigenvalues, eigenvectors) where eigenvalues is (n) and eigenvectors is (n×n)
    pub fn hermitian_eigendecomposition_complex(
        matrix: &Array2<Complex<f64>>,
    ) -> KwaversResult<(Array1<f64>, Array2<Complex<f64>>)> {
        let n = matrix.nrows();
        if matrix.ncols() != n {
            return Err(KwaversError::Numerical(NumericalError::MatrixDimension {
                operation: "hermitian_eigendecomposition_complex".to_string(),
                expected: format!("{}×{} square matrix", n, n),
                actual: format!("{}×{} matrix", matrix.nrows(), matrix.ncols()),
            }));
        }

        // Check if matrix is Hermitian
        for i in 0..n {
            for j in (i + 1)..n {
                if (matrix[[i, j]] - matrix[[j, i]].conj()).norm() > 1e-10 {
                    return Err(KwaversError::Numerical(NumericalError::InvalidOperation(
                        "Matrix must be Hermitian for complex eigendecomposition".to_string(),
                    )));
                }
            }
        }

        // For complex Hermitian matrices, we use a simplified Jacobi-like algorithm
        // In practice, LAPACK would be used for production code
        let mut a = matrix.clone();
        let mut eigenvectors = Array2::eye(n).mapv(|x| Complex::new(x, 0.0));
        let mut eigenvalues = Array1::zeros(n);

        // Simplified complex Jacobi algorithm
        let max_iterations = 50;
        let tolerance = 1e-8;

        for _ in 0..max_iterations {
            let mut max_val = 0.0;
            let mut p = 0;
            let mut q = 1;

            // Find largest off-diagonal element
            for i in 0..n {
                for j in (i + 1)..n {
                    let val = a[[i, j]].norm();
                    if val > max_val {
                        max_val = val;
                        p = i;
                        q = j;
                    }
                }
            }

            if max_val < tolerance {
                break;
            }

            // Compute rotation parameters for complex case
            let a_pp = a[[p, p]].re;
            let a_qq = a[[q, q]].re;
            let a_pq = a[[p, q]];

            let theta = if (a_pp - a_qq).abs() < 1e-12 {
                std::f64::consts::PI / 4.0
            } else {
                0.5 * ((a_qq - a_pp) / a_pq.norm()).atan()
            };

            let c = theta.cos();
            let s = theta.sin();

            // Apply Givens rotation to matrix
            for i in 0..n {
                let a_ip = a[[i, p]];
                let a_iq = a[[i, q]];
                a[[i, p]] = c * a_ip - s * a_iq;
                a[[i, q]] = s * a_ip + c * a_iq;
            }

            for i in 0..n {
                let a_pi = a[[p, i]];
                let a_qi = a[[q, i]];
                a[[p, i]] = c * a_pi - s * a_qi.conj();
                a[[q, i]] = s * a_pi.conj() + c * a_qi;
            }

            // Apply rotation to eigenvectors
            for i in 0..n {
                let v_ip = eigenvectors[[i, p]];
                let v_iq = eigenvectors[[i, q]];
                eigenvectors[[i, p]] = c * v_ip - s * v_iq;
                eigenvectors[[i, q]] = s * v_ip + c * v_iq;
            }
        }

        // Extract real eigenvalues from diagonal
        for i in 0..n {
            eigenvalues[i] = a[[i, i]].re;
        }

        Ok((eigenvalues, eigenvectors))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_real_symmetric_eigendecomposition() {
        // Simple 2×2 symmetric matrix
        let matrix = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0]).unwrap();

        let (eigenvals, eigenvecs) = EigenDecomposition::eigendecomposition(&matrix).unwrap();

        // Check eigenvalues (should be 3 and 1)
        assert!((eigenvals[0] - 3.0).abs() < 1e-6 || (eigenvals[1] - 3.0).abs() < 1e-6);
        assert!((eigenvals[0] - 1.0).abs() < 1e-6 || (eigenvals[1] - 1.0).abs() < 1e-6);

        // Verify A*v = λ*v for each eigenvector
        for i in 0..2 {
            let lambda = eigenvals[i];
            let v = eigenvecs.column(i);
            let av = matrix.dot(&v.to_owned());

            for j in 0..2 {
                assert!((av[j] - lambda * v[j]).abs() < 1e-6);
            }
        }
    }
}