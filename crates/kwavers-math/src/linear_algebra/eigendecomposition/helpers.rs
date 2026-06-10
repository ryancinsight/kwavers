use super::EigenSolver;
use kwavers_core::error::{KwaversError, KwaversResult, NumericalError};
use ndarray::{Array1, Array2};
use num_complex::Complex;

/// `(Q, R)` factor pair from a complex QR decomposition.
type ComplexQr = (Array2<Complex<f64>>, Array2<Complex<f64>>);

impl EigenSolver {
    /// Verify hermitian.
    /// # Errors
    /// - Returns [`KwaversError::Numerical`] if the precondition for a Numerical-class constraint is violated.
    ///
    pub(super) fn verify_hermitian(matrix: &Array2<Complex<f64>>) -> KwaversResult<()> {
        let n = matrix.nrows();
        let tolerance = 1e-10;

        for i in 0..n {
            for j in (i + 1)..n {
                let diff = (matrix[[i, j]] - matrix[[j, i]].conj()).norm();
                if diff > tolerance {
                    return Err(KwaversError::Numerical(NumericalError::InvalidOperation(
                        format!(
                            "Matrix is not Hermitian: ||A[{},{}] - conj(A[{},{}])|| = {:.2e}",
                            i, j, j, i, diff
                        ),
                    )));
                }
            }
        }

        Ok(())
    }
    /// Qr decomposition.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn qr_decomposition(
        matrix: &Array2<Complex<f64>>,
        n: usize,
    ) -> KwaversResult<ComplexQr> {
        let mut h = matrix.clone();
        let mut q = Array2::eye(n).mapv(|x| Complex::new(x, 0.0));

        for k in 0..n.saturating_sub(1) {
            let mut x: Vec<Complex<f64>> = (k..n).map(|i| h[[i, k]]).collect();

            let sigma = x.iter().map(|z| (z.norm()).powi(2)).sum::<f64>().sqrt();
            let sigma = if h[[k, k]].re >= 0.0 { sigma } else { -sigma };

            if sigma.abs() < 1e-14 {
                continue;
            }

            x[0] += Complex::new(sigma, 0.0);
            let x_norm = x.iter().map(|z| (z.norm()).powi(2)).sum::<f64>().sqrt();

            if x_norm.abs() < 1e-14 {
                continue;
            }

            for j in k..n {
                let mut dot = Complex::new(0.0, 0.0);
                for i in k..n {
                    dot += x[i - k].conj() * h[[i, j]];
                }
                let factor = 2.0 * dot / (x_norm * x_norm);
                for i in k..n {
                    h[[i, j]] -= factor * x[i - k];
                }
            }

            for i in 0..n {
                let mut dot = Complex::new(0.0, 0.0);
                for j in k..n {
                    dot += x[j - k].conj() * q[[j, i]];
                }
                let factor = 2.0 * dot / (x_norm * x_norm);
                for j in k..n {
                    q[[j, i]] -= factor * x[j - k];
                }
            }
        }

        Ok((q.mapv(|x| x.conj()), h))
    }

    pub(super) fn wilkinson_shift(matrix: &Array2<Complex<f64>>, n: usize) -> f64 {
        let n_minus_2 = n - 2;
        let a = matrix[[n_minus_2, n_minus_2]].re;
        let b = matrix[[n_minus_2, n - 1]].norm();
        let d = matrix[[n - 1, n - 1]].re;

        let trace = a + d;
        let det = a.mul_add(d, -(b * b));
        let disc = (trace * trace / 4.0 - det).sqrt();

        let lambda1 = trace / 2.0 + disc;
        let lambda2 = trace / 2.0 - disc;

        if (lambda1 - d).abs() < (lambda2 - d).abs() {
            lambda1
        } else {
            lambda2
        }
    }

    pub(super) fn compute_off_diagonal_norm(matrix: &Array2<Complex<f64>>, n: usize) -> f64 {
        let mut norm = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    norm += matrix[[i, j]].norm().powi(2);
                }
            }
        }
        norm.sqrt()
    }
    /// Sort eigenvalues.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub(super) fn sort_eigenvalues(
        eigenvalues: Array1<f64>,
        eigenvectors: Array2<Complex<f64>>,
    ) -> (Array1<f64>, Array2<Complex<f64>>, Vec<usize>) {
        let n = eigenvalues.len();
        let mut indices: Vec<usize> = (0..n).collect();

        indices.sort_by(|&i, &j| eigenvalues[j].total_cmp(&eigenvalues[i]));

        let mut sorted_eigenvalues = Array1::zeros(n);
        let mut sorted_eigenvectors = Array2::zeros((n, n));

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            sorted_eigenvalues[new_idx] = eigenvalues[old_idx];
            for i in 0..n {
                sorted_eigenvectors[[i, new_idx]] = eigenvectors[[i, old_idx]];
            }
        }

        (sorted_eigenvalues, sorted_eigenvectors, indices)
    }
}
