use super::{EigenResult, EigenSolver, EigenSolverConfig};
use kwavers_core::error::{KwaversError, KwaversResult, NumericalError};
use leto::{
    Array1,
    Array2,
};
use eunomia::Complex64;

fn matmul_complex(lhs: &Array2<Complex64>, rhs: &Array2<Complex64>) -> Array2<Complex64> {
    let [rows, inner] = lhs.shape();
    let [rhs_inner, cols] = rhs.shape();
    debug_assert_eq!(inner, rhs_inner);
    let mut out = Array2::from_elem([rows, cols], Complex64::new(0.0, 0.0));
    for i in 0..rows {
        for j in 0..cols {
            let mut sum = Complex64::new(0.0, 0.0);
            for k in 0..inner {
                sum += lhs[[i, k]] * rhs[[k, j]];
            }
            out[[i, j]] = sum;
        }
    }
    out
}

impl EigenSolver {
    /// Compute eigendecomposition of complex Hermitian matrix using QR algorithm
    ///
    /// # Algorithm
    ///
    /// QR algorithm with Wilkinson shift:
    /// 1. Reduce matrix to Hessenberg form
    /// 2. Apply implicit QR iterations with shifts
    /// 3. Extract eigenvalues from diagonal
    /// 4. Back-transform to get eigenvectors
    ///
    /// # Complexity
    ///
    /// - Time: O(n³)
    /// - Space: O(n²)
    /// # Errors
    /// - Returns [`KwaversError::Numerical`] if the precondition for a Numerical-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
pub fn qr_algorithm(
        matrix: &Array2<Complex64>,
        config: EigenSolverConfig,
    ) -> KwaversResult<EigenResult> {
        let n = matrix.shape()[0];

        if matrix.shape()[1] != n {
            return Err(KwaversError::Numerical(NumericalError::MatrixDimension {
                operation: "qr_algorithm".to_owned(),
                expected: format!("{}×{} square matrix", n, n),
                actual: format!("{}×{} matrix", matrix.shape()[0], matrix.shape()[1]),
            }));
        }

        Self::verify_hermitian(matrix)?;

        if n <= 32 {
            return Self::jacobi_hermitian(matrix, config);
        }

        let mut h = matrix.clone();
    let mut q = Array2::eye(n).mapv(|x| Complex64::new(x, 0.0));
        let mut eigenvalues = Array1::zeros([n]);
        let mut iterations = 0;

        for iter in 0..config.max_iterations {
            iterations = iter;

            let shift = if iter % 10 == 0 {
                h[[n - 1, n - 1]].re
            } else {
                Self::wilkinson_shift(&h, n)
            };

            for i in 0..n {
                h[[i, i]] -= Complex64::new(shift, 0.0);
            }

            let (q_iter, r) = Self::qr_decomposition(&h, n)?;
            h = matmul_complex(&r, &q_iter);

            for i in 0..n {
                h[[i, i]] += Complex64::new(shift, 0.0);
            }

            q = matmul_complex(&q, &q_iter);

            let off_diag_norm = Self::compute_off_diagonal_norm(&h, n);
            if off_diag_norm < config.tolerance {
                break;
            }
        }

        for i in 0..n {
            eigenvalues[i] = h[[i, i]].re;
        }

        let (eigenvalues, eigenvectors, _permutation) = if config.sort_descending {
            Self::sort_eigenvalues(eigenvalues, q)
        } else {
            (eigenvalues, q, (0..n).collect())
        };

        let condition_number = if config.estimate_condition {
            if eigenvalues[n - 1].abs() > 1e-14 {
                Some(eigenvalues[0].abs() / eigenvalues[n - 1].abs())
            } else {
                None
            }
        } else {
            None
        };

        let off_diagonal_norm = Self::compute_off_diagonal_norm(&h, n);

        Ok(EigenResult {
            eigenvalues,
            eigenvectors,
            iterations,
            off_diagonal_norm,
            condition_number,
            algorithm: "QR with Wilkinson shift".to_owned(),
        })
    }

    /// Compute eigendecomposition using Jacobi method for Hermitian matrices
    ///
    /// # Algorithm
    ///
    /// Classical Jacobi eigenvalue algorithm:
    /// 1. Find largest off-diagonal element
    /// 2. Apply Givens rotation to zero it out
    /// 3. Repeat until convergence
    ///
    /// # Advantages
    ///
    /// - Guaranteed convergence
    /// - Better numerical stability for ill-conditioned matrices
    ///
    /// # Complexity
    ///
    /// - Time: O(n³) to O(n⁴) depending on matrix
    /// - Space: O(n²)
    /// # Errors
    /// - Returns [`KwaversError::Numerical`] if the precondition for a Numerical-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
pub fn jacobi_hermitian(
        matrix: &Array2<Complex64>,
        config: EigenSolverConfig,
    ) -> KwaversResult<EigenResult> {
        let n = matrix.shape()[0];

        if matrix.shape()[1] != n {
            return Err(KwaversError::Numerical(NumericalError::MatrixDimension {
                operation: "jacobi_hermitian".to_owned(),
                expected: format!("{}×{} square matrix", n, n),
                actual: format!("{}×{} matrix", matrix.shape()[0], matrix.shape()[1]),
            }));
        }

        Self::verify_hermitian(matrix)?;

        let mut h = matrix.clone();
        let mut v = Array2::eye(n).mapv(|x| Complex64::new(x, 0.0));
        let mut eigenvalues = Array1::zeros([n]);
        let mut iterations = 0;

        for sweep in 0..config.max_iterations {
            iterations = sweep;
            let mut max_off_diag: f64 = 0.0;
            let mut _rotations = 0;

            for p in 0..n {
                for q in (p + 1)..n {
                    let h_pp = h[[p, p]].re;
                    let h_qq = h[[q, q]].re;
                    let h_pq = h[[p, q]];
                    let h_pq_norm = h_pq.norm();

                    max_off_diag = max_off_diag.max(h_pq_norm);

                    if h_pq_norm > 1e-15 {
                        let theta = if (h_pp - h_qq).abs() < 1e-12 {
                            std::f64::consts::PI / 4.0
                        } else {
                            0.5 * ((h_qq - h_pp) / (2.0 * h_pq_norm)).atan()
                        };

                        let c = theta.cos();
                        let s = theta.sin();

                        for i in 0..n {
                            if i != p && i != q {
                                let h_ip = h[[i, p]];
                                let h_iq = h[[i, q]];
                                h[[i, p]] = c * h_ip - s * h_iq;
                                h[[i, q]] = s * h_ip + c * h_iq;
                                h[[p, i]] = h[[i, p]].conj();
                                h[[q, i]] = h[[i, q]].conj();
                            }
                        }

                        let h_pp_new =
                            (2.0 * s * c).mul_add(-h_pq.re, (c * c).mul_add(h_pp, s * s * h_qq));
                        let h_qq_new =
                            (2.0 * s * c).mul_add(h_pq.re, (s * s).mul_add(h_pp, c * c * h_qq));

                        h[[p, p]] = Complex64::new(h_pp_new, 0.0);
                        h[[q, q]] = Complex64::new(h_qq_new, 0.0);
                        h[[p, q]] = Complex64::new(0.0, 0.0);
                        h[[q, p]] = Complex64::new(0.0, 0.0);

                        for i in 0..n {
                            let v_ip = v[[i, p]];
                            let v_iq = v[[i, q]];
                            v[[i, p]] = c * v_ip - s * v_iq;
                            v[[i, q]] = s * v_ip + c * v_iq;
                        }

                        _rotations += 1;
                    }
                }
            }

            if max_off_diag < config.tolerance {
                break;
            }
        }

        for i in 0..n {
            eigenvalues[i] = h[[i, i]].re;
        }

        let (eigenvalues, eigenvectors, _) = if config.sort_descending {
            Self::sort_eigenvalues(eigenvalues, v)
        } else {
            (eigenvalues, v, (0..n).collect())
        };

        let condition_number = if config.estimate_condition && eigenvalues[n - 1].abs() > 1e-14 {
            Some(eigenvalues[0].abs() / eigenvalues[n - 1].abs())
        } else {
            None
        };

        let off_diagonal_norm = Self::compute_off_diagonal_norm(&h, n);

        Ok(EigenResult {
            eigenvalues,
            eigenvectors,
            iterations,
            off_diagonal_norm,
            condition_number,
            algorithm: "Jacobi for Hermitian matrices".to_owned(),
        })
    }
}
