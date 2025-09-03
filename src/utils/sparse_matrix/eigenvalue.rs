//! Eigenvalue solvers for sparse matrices
//!
//! References:
//! - Lehoucq et al. (1998): "ARPACK Users Guide"

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
    pub fn inverse_power_iteration(
        &self,
        matrix: &CompressedSparseRowMatrix,
        shift: f64,
    ) -> KwaversResult<(f64, Array1<f64>)> {
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

        // Apply power iteration to (A - shift*I)^(-1)
        // This would require solving linear systems in each iteration
        // For now, return placeholder

        Err(KwaversError::Numerical(NumericalError::NotImplemented {
            feature: "inverse_power_iteration".to_string(),
        }))
    }
}
