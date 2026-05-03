use crate::core::error::{KwaversError, KwaversResult, NumericalError};
use crate::math::linear_algebra::tolerance;
use ndarray::{Array1, Array2};
use num_complex::Complex;

/// Eigenvalue decomposition operations
#[derive(Debug)]
pub struct EigenDecomposition;

impl EigenDecomposition {
    /// Compute eigendecomposition of a real symmetric matrix.
    ///
    /// Uses Jacobi eigenvalue algorithm — suitable for small matrices (< 100×100).
    ///
    /// Returns `(eigenvalues, eigenvectors)` sorted descending.
    pub fn eigendecomposition(matrix: &Array2<f64>) -> KwaversResult<(Array1<f64>, Array2<f64>)> {
        let n = matrix.nrows();
        if matrix.ncols() != n {
            return Err(KwaversError::Numerical(NumericalError::MatrixDimension {
                operation: "eigendecomposition".to_string(),
                expected: format!("{}×{} square matrix", n, n),
                actual: format!("{}×{} matrix", matrix.nrows(), matrix.ncols()),
            }));
        }

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

        let max_iterations = 100;
        let tolerance = 1e-10;

        for _ in 0..max_iterations {
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

            if max_val < tolerance {
                break;
            }

            let theta = if a[[p, p]] == a[[q, q]] {
                std::f64::consts::PI / 4.0
            } else {
                0.5 * (a[[q, q]] - a[[p, p]])
                    / a[[p, q]].atan2((a[[q, q]] - a[[p, p]]) / (2.0 * a[[p, q]]))
            };

            let c = theta.cos();
            let s = theta.sin();

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

            for i in 0..n {
                let v_ip = eigenvectors[[i, p]];
                let v_iq = eigenvectors[[i, q]];
                eigenvectors[[i, p]] = c * v_ip - s * v_iq;
                eigenvectors[[i, q]] = s * v_ip + c * v_iq;
            }
        }

        for i in 0..n {
            eigenvalues[i] = a[[i, i]];
        }

        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());

        let sorted_eigenvals = Array1::from_shape_fn(n, |i| eigenvalues[indices[i]]);
        let sorted_eigenvecs =
            Array2::from_shape_fn((n, n), |(i, j)| eigenvectors[[i, indices[j]]]);

        Ok((sorted_eigenvals, sorted_eigenvecs))
    }

    /// Compute eigendecomposition of a complex Hermitian matrix H ∈ ℂ^(n×n).
    ///
    /// Uses complex Jacobi iteration directly on the Hermitian matrix.
    ///
    /// Returns `(eigenvalues ∈ ℝⁿ, eigenvectors ∈ ℂ^(n×n))` sorted descending.
    ///
    /// ## Algorithm
    ///
    /// Complex Jacobi with Hermitian Givens rotations (Golub & Van Loan 2013, §8.5):
    /// 1. Find largest off-diagonal element |H[p,q]|
    /// 2. Compute complex Givens rotation to zero out H[p,q]
    /// 3. Apply: H ← U† H U
    /// 4. Accumulate: V ← V U
    /// 5. Repeat until convergence
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

        for i in 0..n {
            if matrix[[i, i]].im.abs() > tolerance::HERMITIAN_EIG_TOL {
                return Err(KwaversError::Numerical(NumericalError::InvalidOperation(
                    format!(
                        "Matrix diagonal element [{}] has imaginary part {:.2e}, not Hermitian",
                        i,
                        matrix[[i, i]].im
                    ),
                )));
            }
            for j in (i + 1)..n {
                if (matrix[[i, j]] - matrix[[j, i]].conj()).norm() > tolerance::HERMITIAN_EIG_TOL {
                    return Err(KwaversError::Numerical(NumericalError::InvalidOperation(
                        format!(
                            "Matrix not Hermitian: H[{},{}] = {:?}, H[{},{}]† = {:?}",
                            i,
                            j,
                            matrix[[i, j]],
                            j,
                            i,
                            matrix[[j, i]].conj()
                        ),
                    )));
                }
            }
        }

        let mut h = matrix.clone();
        let mut v = Array2::eye(n).mapv(|x| Complex::new(x, 0.0));

        for _sweep in 0..tolerance::HERMITIAN_EIG_MAX_SWEEPS {
            let mut max_offdiag = 0.0;
            let mut p = 0;
            let mut q = 1;

            for i in 0..n {
                for j in (i + 1)..n {
                    let val = h[[i, j]].norm();
                    if val > max_offdiag {
                        max_offdiag = val;
                        p = i;
                        q = j;
                    }
                }
            }

            if max_offdiag < tolerance::HERMITIAN_EIG_TOL {
                break;
            }

            let h_pp = h[[p, p]].re;
            let h_qq = h[[q, q]].re;
            let h_pq = h[[p, q]];

            let theta = if (h_pp - h_qq).abs() < tolerance::HERMITIAN_EIG_TOL {
                std::f64::consts::FRAC_PI_4
            } else {
                0.5 * (2.0 * h_pq.norm() / (h_pp - h_qq)).atan()
                    * if h_pp > h_qq { 1.0 } else { -1.0 }
            };

            let c = theta.cos();
            let s_mag = theta.sin();

            let phase = if h_pq.norm() > tolerance::HERMITIAN_EIG_TOL {
                h_pq.conj() / h_pq.norm()
            } else {
                Complex::new(1.0, 0.0)
            };

            let s_complex = Complex::new(s_mag, 0.0) * phase;

            // Apply H ← H U (right multiply)
            for i in 0..n {
                let h_ip = h[[i, p]];
                let h_iq = h[[i, q]];
                h[[i, p]] = c * h_ip - s_complex.conj() * h_iq;
                h[[i, q]] = s_complex * h_ip + c * h_iq;
            }

            // Apply H ← U† H (left multiply)
            for j in 0..n {
                let h_pj = h[[p, j]];
                let h_qj = h[[q, j]];
                h[[p, j]] = c * h_pj - s_complex * h_qj;
                h[[q, j]] = s_complex.conj() * h_pj + c * h_qj;
            }

            // Accumulate eigenvectors V ← V U
            for i in 0..n {
                let v_ip = v[[i, p]];
                let v_iq = v[[i, q]];
                v[[i, p]] = c * v_ip - s_complex.conj() * v_iq;
                v[[i, q]] = s_complex * v_ip + c * v_iq;
            }
        }

        let mut eigenvals = Array1::zeros(n);
        for i in 0..n {
            eigenvals[i] = h[[i, i]].re;
        }

        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| eigenvals[j].partial_cmp(&eigenvals[i]).unwrap());

        let sorted_eigenvals = Array1::from_shape_fn(n, |i| eigenvals[indices[i]]);
        let sorted_eigenvecs = Array2::from_shape_fn((n, n), |(i, j)| v[[i, indices[j]]]);

        Ok((sorted_eigenvals, sorted_eigenvecs))
    }
}
