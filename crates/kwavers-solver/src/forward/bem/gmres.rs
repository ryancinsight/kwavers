//! Preconditioned GMRES(restart) for dense real linear systems.
//!
//! # Algorithm (Saad & Schultz 1986, §2)
//!
//! Restarted GMRES with Modified Gram-Schmidt Arnoldi and Givens rotations
//! for the least-squares subproblem. Uses left Jacobi (diagonal) preconditioning:
//! `M = diag(A)`, solving `M⁻¹A x = M⁻¹b`.
//!
//! ## Convergence guarantee
//!
//! **Theorem** (Saad & Schultz 1986, Theorem 2.1): In exact arithmetic, GMRES
//! converges in at most N outer iterations (each of length `restart`) for an
//! N×N non-singular system. With diagonal preconditioning, convergence is faster
//! when diagonal dominance is high (common in BEM H matrices: `H[i,i] ≈ 0.5`).
//!
//! ## Memory
//!
//! O(N · restart) per outer restart cycle.
//!
//! # References
//!
//! - Saad, Y. & Schultz, M.H. (1986). "GMRES: A generalized minimal residual
//!   algorithm for solving nonsymmetric linear systems." SIAM J. Sci. Stat.
//!   Comput. 7(3), 856–869. DOI: 10.1137/0907058

use kwavers_core::error::{KwaversError, KwaversResult, NumericalError};
use ndarray::{Array1, Array2};

/// Solve `A·x = b` using preconditioned restarted GMRES.
///
/// # Arguments
///
/// * `a`        — Dense N×N coefficient matrix
/// * `rhs`      — Right-hand side vector, length N
/// * `tol`      — Relative residual tolerance: ‖r‖₂ / ‖b‖₂ < tol
/// * `max_iter` — Maximum outer restarts
/// * `restart`  — Krylov subspace dimension per restart (typical: 20–50)
///
/// # Errors
///
/// Returns `KwaversError::Numerical(ConvergenceFailed)` if convergence is not
/// achieved within `max_iter * restart` matrix-vector products.
/// # Panics
/// - Panics if an internal precondition is violated.
///
pub fn solve_gmres(
    a: &Array2<f64>,
    rhs: &Array1<f64>,
    tol: f64,
    max_iter: usize,
    restart: usize,
) -> KwaversResult<Array1<f64>> {
    let n = rhs.len();
    debug_assert_eq!(
        a.shape(),
        &[n, n],
        "A must be N×N square; got {:?}",
        a.shape()
    );

    // Jacobi (diagonal) preconditioner: d[i] = 1/A[i,i]
    let precond: Vec<f64> = (0..n)
        .map(|i| {
            let aii = a[[i, i]];
            if aii.abs() < 1e-300 {
                1.0
            } else {
                1.0 / aii
            }
        })
        .collect();

    // Preconditioned RHS norm for relative tolerance
    let pb_norm = {
        let mut s = 0.0_f64;
        for i in 0..n {
            s += (precond[i] * rhs[i]).powi(2);
        }
        s.sqrt()
    };
    if pb_norm < 1e-300 {
        return Ok(Array1::zeros(n));
    }

    let mut x = Array1::<f64>::zeros(n);

    for _outer in 0..max_iter {
        // r = M⁻¹(b − Ax)
        let ax = a.dot(&x);
        let mut r = Array1::<f64>::zeros(n);
        for i in 0..n {
            r[i] = precond[i] * (rhs[i] - ax[i]);
        }

        let beta = r.dot(&r).sqrt();
        if beta / pb_norm < tol {
            return Ok(x);
        }

        // Krylov basis V (columns), upper Hessenberg H
        let m = restart;
        let mut v = Array2::<f64>::zeros((n, m + 1));
        let mut h = Array2::<f64>::zeros((m + 1, m));
        let mut c_rot = vec![0.0_f64; m];
        let mut s_rot = vec![0.0_f64; m];
        let mut g = Array1::<f64>::zeros(m + 1);
        g[0] = beta;

        for i in 0..n {
            v[[i, 0]] = r[i] / beta;
        }

        let mut j_end = m;
        for j in 0..m {
            // w = M⁻¹ · A · v[:,j]
            let v_j: Array1<f64> = v.column(j).to_owned();
            let av_j = a.dot(&v_j);
            let mut w = Array1::<f64>::zeros(n);
            for i in 0..n {
                w[i] = precond[i] * av_j[i];
            }

            // Modified Gram-Schmidt
            for i in 0..=j {
                let v_i = v.column(i);
                let hij: f64 = v_i.iter().zip(w.iter()).map(|(&a, &b)| a * b).sum();
                h[[i, j]] = hij;
                for k in 0..n {
                    w[k] -= hij * v_i[k];
                }
            }
            let h_jp1 = w.dot(&w).sqrt();
            h[[j + 1, j]] = h_jp1;

            // Apply previous Givens rotations
            for i in 0..j {
                let hi = h[[i, j]];
                let hi1 = h[[i + 1, j]];
                h[[i, j]] = c_rot[i].mul_add(hi, s_rot[i] * hi1);
                h[[i + 1, j]] = (-s_rot[i]).mul_add(hi, c_rot[i] * hi1);
            }

            // New Givens rotation
            let denom = h[[j, j]].hypot(h[[j + 1, j]]);
            if denom < 1e-300 {
                j_end = j + 1;
                break;
            }
            c_rot[j] = h[[j, j]] / denom;
            s_rot[j] = h[[j + 1, j]] / denom;
            h[[j, j]] = denom;
            h[[j + 1, j]] = 0.0;

            let gj = g[j];
            g[j] = c_rot[j] * gj;
            g[j + 1] = -s_rot[j] * gj;

            // Next basis vector
            if h_jp1 > 1e-300 && j + 1 < m {
                for i in 0..n {
                    v[[i, j + 1]] = w[i] / h_jp1;
                }
            }

            // Convergence check
            if g[j + 1].abs() / pb_norm < tol {
                j_end = j + 1;
                break;
            }
        }

        // Back substitution: H[0..j_end, 0..j_end] · y = g[0..j_end]
        let mut y = Array1::<f64>::zeros(j_end);
        for i in (0..j_end).rev() {
            y[i] = g[i];
            for k in (i + 1)..j_end {
                y[i] -= h[[i, k]] * y[k];
            }
            if h[[i, i]].abs() > 1e-300 {
                y[i] /= h[[i, i]];
            }
        }

        // Update solution: x += V[:,0..j_end] · y
        for k in 0..j_end {
            let v_k = v.column(k).to_owned();
            for i in 0..n {
                x[i] += y[k] * v_k[i];
            }
        }

        // Final convergence check
        let r_final: Array1<f64> = rhs - &a.dot(&x);
        let r_norm: f64 = r_final.dot(&r_final).sqrt();
        let b_norm: f64 = rhs.dot(rhs).sqrt().max(1e-300);
        if r_norm / b_norm < tol {
            return Ok(x);
        }
    }

    // Non-convergence
    let r_final: Array1<f64> = rhs - &a.dot(&x);
    let final_res = r_final.dot(&r_final).sqrt() / rhs.dot(rhs).sqrt().max(1e-300);
    Err(KwaversError::Numerical(NumericalError::ConvergenceFailed {
        method: "GMRES".to_owned(),
        iterations: max_iter * restart,
        error: final_res,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GMRES must recover the exact solution of a well-conditioned 5×5 system.
    ///
    /// **Theorem** (Saad & Schultz 1986, Theorem 2.1): exact convergence in ≤N steps.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_gmres_matches_direct_small_system() {
        let n = 5usize;
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            a[[i, i]] = 10.0 + i as f64;
            if i > 0 {
                a[[i, i - 1]] = -1.0;
            }
            if i < n - 1 {
                a[[i, i + 1]] = -1.0;
            }
        }
        let x_true = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let rhs = a.dot(&x_true);

        let x_gmres = solve_gmres(&a, &rhs, 1e-12, 20, 10).unwrap();

        let max_err = x_gmres
            .iter()
            .zip(x_true.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err < 1e-10,
            "GMRES solution error too large: {max_err:.3e}"
        );
    }

    /// GMRES must converge within N iterations for an N×N non-singular system.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_gmres_converges_within_n_iters() {
        let n = 8usize;
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            a[[i, i]] = 4.0;
            if i > 0 {
                a[[i, i - 1]] = -1.0;
            }
            if i < n - 1 {
                a[[i, i + 1]] = -1.0;
            }
        }
        let rhs = Array1::from_elem(n, 1.0);

        let x = solve_gmres(&a, &rhs, 1e-12, 1, n).unwrap();

        let residual: Array1<f64> = rhs.clone() - a.dot(&x);
        let res_norm = residual.dot(&residual).sqrt();
        let rhs_norm = rhs.dot(&rhs).sqrt();
        assert!(
            res_norm / rhs_norm < 1e-10,
            "GMRES residual too large: rel={:.3e}",
            res_norm / rhs_norm
        );
    }

    /// GMRES with non-converging tolerance must return ConvergenceFailed error.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[test]
    fn test_gmres_nonconvergence_returns_error() {
        let n = 3usize;
        let a = Array2::<f64>::zeros((n, n));
        let rhs = Array1::from_elem(n, 1.0);
        let result = solve_gmres(&a, &rhs, 1e-14, 2, 3);
        assert!(result.is_err(), "GMRES should fail on singular system");
    }
}
