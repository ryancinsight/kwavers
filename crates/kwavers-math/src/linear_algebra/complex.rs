use kwavers_core::error::{KwaversError, KwaversResult, NumericalError};
use leto::{Array1, Array2, Storage};
use num_complex::Complex;

/// Complex linear algebra operations for beamforming
#[derive(Debug)]
pub struct ComplexLinearAlgebra;

impl ComplexLinearAlgebra {
    /// Solve a complex linear system Ax = b using Gaussian elimination with partial pivoting.
    pub fn solve_linear_system_complex(
        a: &Array2<Complex<f64>>,
        b: &Array1<Complex<f64>>,
    ) -> KwaversResult<Array1<Complex<f64>>> {
        let n = a.shape()[0];
        if a.shape()[1] != n || b.shape()[0] != n {
            return Err(KwaversError::Numerical(NumericalError::MatrixDimension {
                operation: "solve_linear_system_complex".to_owned(),
                expected: format!("{n}×{n} matrix and {n} vector"),
                actual: format!(
                    "{}×{} matrix and {} vector",
                    a.shape()[0],
                    a.shape()[1],
                    b.shape()[0]
                ),
            }));
        }

        let mut a_data = a.storage().as_slice().to_vec();
        let mut b_data = b.storage().as_slice().to_vec();

        // Forward elimination
        for i in 0..n {
            // Find pivot (largest magnitude)
            let mut max_row = i;
            let mut max_val = a_data[i * n + i].norm_sqr();

            for k in (i + 1)..n {
                let val = a_data[k * n + i].norm_sqr();
                if val > max_val {
                    max_val = val;
                    max_row = k;
                }
            }

            // Swap rows if needed
            if max_row != i {
                for j in 0..n {
                    a_data.swap(i * n + j, max_row * n + j);
                }
                b_data.swap(i, max_row);
            }

            // Check for singularity
            if a_data[i * n + i].norm_sqr() < 1e-24 {
                return Err(KwaversError::Numerical(NumericalError::SingularMatrix {
                    operation: "solve_linear_system_complex".to_owned(),
                    condition_number: f64::INFINITY,
                }));
            }

            let pivot = a_data[i * n + i];

            // Eliminate below
            for k in (i + 1)..n {
                let factor = a_data[k * n + i] / pivot;
                for j in i..n {
                    let v = a_data[k * n + j];
                    a_data[k * n + j] = v - factor * a_data[i * n + j];
                }
                b_data[k] = b_data[k] - factor * b_data[i];
            }
        }

        // Back substitution
        let mut x = vec![Complex::new(0.0, 0.0); n];
        for i in (0..n).rev() {
            let mut sum = Complex::new(0.0, 0.0);
            for j in (i + 1)..n {
                sum += a_data[i * n + j] * x[j];
            }
            x[i] = (b_data[i] - sum) / a_data[i * n + i];
        }

        Ok(Array1::from(x))
    }

    /// Compute inverse of a complex matrix via column-wise solves.
    pub fn matrix_inverse_complex(
        matrix: &Array2<Complex<f64>>,
    ) -> KwaversResult<Array2<Complex<f64>>> {
        let n = matrix.shape()[0];
        if matrix.shape()[1] != n {
            return Err(KwaversError::Numerical(NumericalError::MatrixDimension {
                operation: "matrix_inverse_complex".to_owned(),
                expected: format!("{n}×{n} square matrix"),
                actual: format!("{}×{} matrix", matrix.shape()[0], matrix.shape()[1]),
            }));
        }

        let mut result = vec![Complex::new(0.0, 0.0); n * n];

        // Solve for each column of the identity matrix
        for col in 0..n {
            let mut e = vec![Complex::new(0.0, 0.0); n];
            e[col] = Complex::new(1.0, 0.0);
            let e_array = Array1::from(e);
            let x = Self::solve_linear_system_complex(matrix, &e_array)?;
            let xs = x.storage().as_slice();
            for row in 0..n {
                result[row * n + col] = xs[row];
            }
        }

        Array2::from_shape_vec([n, n], result)
            .map_err(|e| KwaversError::Numerical(NumericalError::SolverFailed {
                method: "matrix_inverse_complex".to_owned(),
                reason: e.to_string(),
            }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_complex_linear_system() {
        let a = Array2::from_shape_vec(
            [2, 2],
            vec![
                Complex::new(2.0, 1.0),
                Complex::new(1.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(2.0, -1.0),
            ],
        )
        .unwrap();

        let b = Array1::from(vec![Complex::new(3.0, 1.0), Complex::new(2.0, -1.0)]);

        let x = ComplexLinearAlgebra::solve_linear_system_complex(&a, &b).unwrap();

        // Verify Ax = b
        let a_sl = a.storage().as_slice();
        let x_sl = x.storage().as_slice();
        let b_sl = b.storage().as_slice();
        for i in 0..2 {
            let mut sum = Complex::new(0.0, 0.0);
            for j in 0..2 {
                sum = sum + a_sl[i * 2 + j] * x_sl[j];
            }
            assert!((sum - b_sl[i]).norm() < 1e-10);
        }
    }
}
