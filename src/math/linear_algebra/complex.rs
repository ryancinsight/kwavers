//! Complex Linear Algebra Operations
//!
//! This module provides linear algebra operations for complex-valued matrices
//! and vectors, essential for narrowband beamforming algorithms like MVDR/Capon.

use crate::core::error::{KwaversError, KwaversResult, NumericalError};
use ndarray::{Array1, Array2};
use num_complex::Complex;

/// Complex linear algebra operations for beamforming
#[derive(Debug)]
pub struct ComplexLinearAlgebra;

impl ComplexLinearAlgebra {
    /// Solve a complex linear system Ax = b
    ///
    /// # Arguments
    /// * `a` - Complex coefficient matrix (n×n)
    /// * `b` - Complex right-hand side vector (n)
    ///
    /// # Returns
    /// Complex solution vector x
    pub fn solve_linear_system_complex(
        a: &Array2<Complex<f64>>,
        b: &Array1<Complex<f64>>,
    ) -> KwaversResult<Array1<Complex<f64>>> {
        let n = a.nrows();
        if a.ncols() != n || b.len() != n {
            return Err(KwaversError::Numerical(NumericalError::MatrixDimension {
                operation: "solve_linear_system_complex".to_string(),
                expected: format!("{}×{} matrix and {} vector", n, n, n),
                actual: format!("{}×{} matrix and {} vector", a.nrows(), a.ncols(), b.len()),
            }));
        }

        // Gaussian elimination with partial pivoting for complex matrices
        let mut a_copy = a.clone();
        let mut b_copy = b.clone();

        // Forward elimination
        for i in 0..n {
            // Find pivot (largest magnitude)
            let mut max_row = i;
            let mut max_val = a_copy[[i, i]].norm();

            for k in (i + 1)..n {
                let val = a_copy[[k, i]].norm();
                if val > max_val {
                    max_val = val;
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
            }

            // Check for singularity
            if a_copy[[i, i]].norm() < 1e-12 {
                return Err(KwaversError::Numerical(NumericalError::SingularMatrix {
                    operation: "solve_linear_system_complex".to_string(),
                    condition_number: f64::INFINITY,
                }));
            }

            let pivot = a_copy[[i, i]];
            let pivot_row = a_copy.row(i).to_owned();
            let pivot_b = b_copy[i];

            // Eliminate
            for k in (i + 1)..n {
                let factor = a_copy[[k, i]] / pivot;
                for j in i..n {
                    a_copy[[k, j]] -= factor * pivot_row[j];
                }
                b_copy[k] -= factor * pivot_b;
            }
        }

        // Back substitution
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            let mut sum = Complex::new(0.0, 0.0);
            for j in (i + 1)..n {
                sum += a_copy[[i, j]] * x[j];
            }
            x[i] = (b_copy[i] - sum) / a_copy[[i, i]];
        }

        Ok(x)
    }

    /// Compute inverse of a complex matrix
    ///
    /// # Arguments
    /// * `matrix` - Complex square matrix (n×n)
    ///
    /// # Returns
    /// Complex inverse matrix
    pub fn matrix_inverse_complex(
        matrix: &Array2<Complex<f64>>,
    ) -> KwaversResult<Array2<Complex<f64>>> {
        let n = matrix.nrows();
        if matrix.ncols() != n {
            return Err(KwaversError::Numerical(NumericalError::MatrixDimension {
                operation: "matrix_inverse_complex".to_string(),
                expected: format!("{}×{} square matrix", n, n),
                actual: format!("{}×{} matrix", matrix.nrows(), matrix.ncols()),
            }));
        }

        // Create identity matrix
        let identity = Array2::eye(n).mapv(|x| Complex::new(x, 0.0));
        let mut result = Array2::zeros((n, n));

        // Solve for each column of the identity matrix
        for i in 0..n {
            let b = identity.column(i).to_owned();
            let x = Self::solve_linear_system_complex(matrix, &b)?;
            result.column_mut(i).assign(&x);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use num_complex::Complex;

    #[test]
    fn test_solve_complex_linear_system() {
        // Simple 2×2 complex system
        let a = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex::new(2.0, 1.0),
                Complex::new(1.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(2.0, -1.0),
            ],
        )
        .unwrap();

        let b = Array1::from_vec(vec![Complex::new(3.0, 1.0), Complex::new(2.0, -1.0)]);

        let x = ComplexLinearAlgebra::solve_linear_system_complex(&a, &b).unwrap();

        // Verify solution by checking Ax = b
        let ax = a.dot(&x);
        for i in 0..2 {
            assert!((ax[i] - b[i]).norm() < 1e-10);
        }
    }
}
