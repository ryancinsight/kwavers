//! Eigenvalue solvers for sparse matrices
//!
//! References:
//! - Lehoucq et al. (1998): "ARPACK Users Guide"
//! - Golub & Van Loan (2013): "Matrix Computations"

use super::csr::CompressedSparseRowMatrix;
use crate::error::{KwaversError, KwaversResult, NumericalError};
use ndarray::Array1;

/// Eigenvalue solver for sparse matrices
#[derive(Debug)]
pub struct EigenvalueSolver {
    /// Maximum iterations
    max_iterations: usize,
    /// Convergence tolerance
    tolerance: f64,
}

impl EigenvalueSolver {
    /// Create eigenvalue solver
    #[must_use]
    pub fn create(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }

    /// Power iteration for largest eigenvalue
    pub fn power_iteration(
        &self,
        matrix: &CompressedSparseRowMatrix,
    ) -> KwaversResult<(f64, Array1<f64>)> {
        if matrix.rows != matrix.cols {
            return Err(KwaversError::Numerical(NumericalError::Instability {
                operation: "eigenvalue_solver".to_string(),
                condition: matrix.rows as f64,
            }));
        }

        let n = matrix.rows;
        let mut v = Array1::from_elem(n, 1.0 / (n as f64).sqrt());
        let mut eigenvalue = 0.0;

        for iteration in 0..self.max_iterations {
            let v_prev = v.clone();

            // v = A * v
            v = matrix.multiply_vector(v.view())?;

            // Normalize
            let norm = v.dot(&v).sqrt();
            if norm < 1e-14 {
                return Err(KwaversError::Numerical(NumericalError::Instability {
                    operation: "power_iteration".to_string(),
                    condition: norm,
                }));
            }
            v /= norm;

            // Rayleigh quotient
            let av = matrix.multiply_vector(v.view())?;
            eigenvalue = v.dot(&av);

            // Check convergence
            let diff = (&v - &v_prev).dot(&(&v - &v_prev)).sqrt();
            if diff < self.tolerance {
                return Ok((eigenvalue, v));
            }

            if iteration % 10 == 0 {
                log::debug!(
                    "Power iteration {}: eigenvalue = {:.6}, residual = {:.2e}",
                    iteration,
                    eigenvalue,
                    diff
                );
            }
        }

        Err(KwaversError::Numerical(NumericalError::ConvergenceFailed {
            method: "power_iteration".to_string(),
            iterations: self.max_iterations,
            error: eigenvalue,
        }))
    }

    /// Inverse power iteration for smallest eigenvalue
    /// Reference: Golub & Van Loan (2013) "Matrix Computations", Algorithm 7.3.3
    pub fn inverse_power_iteration(
        &self,
        matrix: &CompressedSparseRowMatrix,
        shift: f64,
    ) -> KwaversResult<(f64, Array1<f64>)> {
        let n = matrix.rows;
        
        // Create shifted matrix: A - shift*I
        let mut shifted = matrix.clone();
        for i in 0..matrix.rows {
            for j in shifted.row_pointers[i]..shifted.row_pointers[i + 1] {
                if shifted.col_indices[j] == i {
                    shifted.values[j] -= shift;
                    break;
                }
            }
        }

        // Initial guess - random vector
        let mut v = Array1::from(
            (0..n).map(|_| rand::random::<f64>() - 0.5).collect::<Vec<_>>()
        );
        
        // Normalize
        let norm = v.dot(&v).sqrt();
        v /= norm;

        let mut eigenvalue = 0.0;
        
        // Inverse power iteration
        for _iter in 0..self.max_iterations {
            // Solve (A - shift*I) * w = v using simple iterative method
            // For production use, should implement LU decomposition or GMRES
            let mut w = Array1::zeros(n);
            
            // Simple Jacobi iteration for linear solve (A - shift*I)w = v
            for _jacobi in 0..50 {
                let mut w_new = Array1::zeros(n);
                
                for i in 0..n {
                    let mut sum = v[i];
                    let mut diagonal = 1.0; // Default diagonal element
                    
                    for j in shifted.row_pointers[i]..shifted.row_pointers[i + 1] {
                        let col = shifted.col_indices[j];
                        let val = shifted.values[j];
                        
                        if col == i {
                            diagonal = val;
                        } else {
                            sum -= val * w[col];
                        }
                    }
                    
                    if diagonal.abs() > 1e-14 {
                        w_new[i] = sum / diagonal;
                    }
                }
                
                w = w_new;
            }
            
            // Rayleigh quotient: λ = v^T * A * w / (v^T * w)
            let av = self.matrix_vector_multiply(matrix, &w)?;
            let numerator = v.dot(&av);
            let denominator = v.dot(&w);
            
            if denominator.abs() < 1e-14 {
                return Err(KwaversError::Numerical(NumericalError::ConvergenceFailure {
                    method: "inverse_power_iteration".to_string(),
                    iterations: self.max_iterations,
                    residual: denominator.abs(),
                }));
            }
            
            let lambda = numerator / denominator;
            
            // Check convergence
            if (lambda - eigenvalue).abs() < self.tolerance {
                // Normalize eigenvector
                let norm = w.dot(&w).sqrt();
                if norm > 1e-14 {
                    w /= norm;
                }
                return Ok((lambda, w));
            }
            
            eigenvalue = lambda;
            
            // Normalize for next iteration
            let norm = w.dot(&w).sqrt();
            if norm > 1e-14 {
                v = w / norm;
            } else {
                return Err(KwaversError::Numerical(NumericalError::ConvergenceFailure {
                    method: "inverse_power_iteration".to_string(),
                    iterations: _iter + 1,
                    residual: norm,
                }));
            }
        }

        Err(KwaversError::Numerical(NumericalError::ConvergenceFailure {
            method: "inverse_power_iteration".to_string(),
            iterations: self.max_iterations,
            residual: self.tolerance,
        }))
    }
    
    /// Helper method for matrix-vector multiplication
    fn matrix_vector_multiply(
        &self,
        matrix: &CompressedSparseRowMatrix,
        x: &Array1<f64>,
    ) -> KwaversResult<Array1<f64>> {
        let mut result = Array1::zeros(matrix.rows);
        
        for i in 0..matrix.rows {
            for j in matrix.row_pointers[i]..matrix.row_pointers[i + 1] {
                let col = matrix.col_indices[j];
                let val = matrix.values[j];
                result[i] += val * x[col];
            }
        }
        
        Ok(result)
    }
}
