//! Jacobi-preconditioned GMRES for the dense boundary-element system.
//!
//! The recurrence is Athena's (Atlas ADR 0033). This module supplies only the
//! boundary-element system's own vocabulary: it borrows the assembled dense
//! coefficient block as an Athena
//! [`BorrowedDenseOperator`], derives the diagonal preconditioner the BEM `H`
//! matrix rewards, and reports a non-converged solve as a kwavers numerical
//! error rather than a partial iterate.
//!
//! # Why diagonal preconditioning
//!
//! Burton-Miller BEM `H` matrices are diagonally dominant (`H[i,i] ≈ 0.5`), so
//! scaling rows by the inverse diagonal clusters the spectrum and shortens the
//! Krylov subspace GMRES needs. Athena applies the preconditioner on the right,
//! solving `A M⁻¹ y = b` with `x = M⁻¹ y`; the residual it reports is therefore
//! the true residual `‖b − A x‖`, not a preconditioned surrogate.
//!
//! # References
//!
//! - Saad, Y. & Schultz, M.H. (1986). "GMRES: A generalized minimal residual
//!   algorithm for solving nonsymmetric linear systems." SIAM J. Sci. Stat.
//!   Comput. 7(3), 856–869. DOI: 10.1137/0907058

use crate::krylov::{convergence_failure, KrylovWorkspace};
use athena_leto::{BorrowedDenseOperator, Jacobi};
use kwavers_core::error::{KwaversError, KwaversResult, NumericalError};
use leto::{Array1, Array2};
use leto_ops::CsrMatrix;

/// Solve `A·x = b` with Jacobi-preconditioned restarted GMRES.
///
/// # Arguments
///
/// * `a`        — Dense N×N coefficient matrix
/// * `rhs`      — Right-hand side vector, length N
/// * `tol`      — Relative residual tolerance: ‖r‖₂ / ‖b‖₂ < tol
/// * `max_iter` — Maximum outer restart cycles
/// * `restart`  — Krylov subspace dimension per cycle (typical: 20–50)
///
/// # Errors
///
/// Returns [`NumericalError::ConvergenceFailed`] when the solve does not reach
/// `tol` within `max_iter * restart` operator applications, and
/// [`NumericalError::MatrixDimension`] when `a` is not `rhs.len()` square.
/// Returns a backend failure when the iteration budget overflows or the
/// restart width is zero.
///
/// # Panics
///
/// Panics if the budget or policy construction receives a zero restart width;
/// both are validated first and surfaced as typed errors instead.
pub fn solve_gmres(
    a: &Array2<f64>,
    rhs: &Array1<f64>,
    tol: f64,
    max_iter: usize,
    restart: usize,
) -> KwaversResult<Array1<f64>> {
    let dimension = rhs.len();
    let shape = a.shape();
    if shape != [dimension, dimension] {
        return Err(KwaversError::Numerical(NumericalError::MatrixDimension {
            operation: "BEM GMRES".to_owned(),
            expected: format!("[{dimension}, {dimension}]"),
            actual: format!("{shape:?}"),
        }));
    }

    // Athena's dense operator borrows row-major coefficients. A strided or
    // transposed assembly is materialized once here rather than per operator
    // application.
    let materialized;
    let coefficients = match a.as_slice() {
        Some(slice) => slice,
        None => {
            materialized = a.to_contiguous();
            materialized
                .as_slice()
                .expect("invariant: materialized array is C-contiguous")
        }
    };
    let operator = BorrowedDenseOperator::new(dimension, coefficients)
        .map_err(|error| backend_failure("BEM dense operator", &error))?;
    let preconditioner = jacobi_from_diagonal(a, dimension)?;

    let right_hand_side = contiguous(rhs);
    let mut solution = Array1::<f64>::zeros([dimension]);
    let policy = crate::krylov::policy(0.0, tol, budget(max_iter, restart)?)?;
    let mut workspace = KrylovWorkspace::new(restart, dimension)?;

    let report = workspace.solve(
        &operator,
        &preconditioner,
        &right_hand_side,
        &mut solution,
        policy,
    )?;
    if report.converged() {
        Ok(solution)
    } else {
        Err(convergence_failure("GMRES", &report))
    }
}

/// Total operator applications the restarted solve may spend.
fn budget(max_iter: usize, restart: usize) -> KwaversResult<usize> {
    max_iter.checked_mul(restart).ok_or_else(|| {
        KwaversError::Numerical(NumericalError::InvalidOperation(format!(
            "BEM GMRES budget {max_iter} restarts x {restart} Krylov dimensions overflows usize"
        )))
    })
}

/// Build the Jacobi preconditioner from the dense matrix diagonal.
///
/// Athena derives the inverse diagonal from a sparse matrix, and the diagonal
/// of a dense operator is exactly a diagonal CSR matrix — `N` stored entries
/// beside the `N²` the dense block already holds.
///
/// A diagonal entry that is not a normal float leaves its row unscaled. For any
/// normal `x`, `1/x` is finite (`1/f64::MIN_POSITIVE ≈ 4.5e307 < f64::MAX`),
/// while zero, subnormal, infinite, and NaN entries have no usable reciprocal;
/// substituting one for them is the standard Jacobi guard and keeps the
/// preconditioner the identity on those rows rather than poisoning the solve.
fn jacobi_from_diagonal(a: &Array2<f64>, dimension: usize) -> KwaversResult<Jacobi<f64>> {
    let mut values = Vec::with_capacity(dimension);
    for index in 0..dimension {
        let entry = a[[index, index]];
        values.push(if entry.is_normal() { entry } else { 1.0 });
    }
    let matrix = CsrMatrix::from_parts(
        values,
        (0..dimension).collect(),
        (0..=dimension).collect(),
        dimension,
        dimension,
    )
    .map_err(|error| {
        KwaversError::Numerical(NumericalError::SolverFailed {
            method: "BEM Jacobi preconditioner".to_owned(),
            reason: error.to_string(),
        })
    })?;
    Jacobi::from_csr(&matrix).map_err(|error| backend_failure("BEM Jacobi preconditioner", &error))
}

/// Borrow `vector` as a dense row-major array, materializing a strided one.
fn contiguous(vector: &Array1<f64>) -> Array1<f64> {
    if vector.as_slice().is_some() {
        vector.clone()
    } else {
        vector.to_contiguous()
    }
}

fn backend_failure(operation: &str, error: &athena_leto::LetoBackendError) -> KwaversError {
    KwaversError::Numerical(NumericalError::SolverFailed {
        method: operation.to_owned(),
        reason: error.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Relative residual `‖b − A·x‖₂ / ‖b‖₂`.
    fn relative_residual(a: &Array2<f64>, rhs: &Array1<f64>, x: &Array1<f64>) -> f64 {
        let n = rhs.len();
        let mut image = Array1::<f64>::zeros([n]);
        leto_ops::matvec(&a.view(), &x.view(), &mut image.view_mut())
            .expect("invariant: residual A·x shapes conform");
        let residual_norm = rhs
            .iter()
            .zip(image.iter())
            .map(|(&b, &ax)| (b - ax).powi(2))
            .sum::<f64>()
            .sqrt();
        let rhs_norm = rhs.iter().map(|&b| b * b).sum::<f64>().sqrt();
        residual_norm / rhs_norm.max(f64::MIN_POSITIVE)
    }

    /// GMRES must recover the exact solution of a well-conditioned 5×5 system.
    ///
    /// **Theorem** (Saad & Schultz 1986, Theorem 2.1): exact convergence in ≤N steps.
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
        let x_true = Array1::from_vec(5, vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let mut rhs = Array1::<f64>::zeros(n);
        leto_ops::matvec(&a.view(), &x_true.view(), &mut rhs.view_mut()).unwrap();

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

        let relative = relative_residual(&a, &rhs, &x);
        assert!(
            relative < 1e-10,
            "GMRES residual too large: rel={relative:.3e}"
        );
    }

    /// GMRES with non-converging tolerance must return ConvergenceFailed error.
    #[test]
    fn test_gmres_nonconvergence_returns_error() {
        let n = 3usize;
        let a = Array2::<f64>::zeros((n, n));
        let rhs = Array1::from_elem(n, 1.0);
        let result = solve_gmres(&a, &rhs, 1e-14, 2, 3);
        assert!(
            matches!(
                result,
                Err(KwaversError::Numerical(
                    NumericalError::ConvergenceFailed { .. }
                ))
            ),
            "GMRES should report convergence failure on a singular system, got {result:?}"
        );
    }

    /// A mismatched right-hand side is rejected before any solve is attempted.
    #[test]
    fn test_gmres_rejects_non_square_system() {
        let a = Array2::<f64>::zeros((3, 3));
        let rhs = Array1::from_elem(4, 1.0);
        let result = solve_gmres(&a, &rhs, 1e-10, 2, 3);
        assert!(
            matches!(
                result,
                Err(KwaversError::Numerical(
                    NumericalError::MatrixDimension { .. }
                ))
            ),
            "a 3x3 matrix must not accept a length-4 right-hand side, got {result:?}"
        );
    }
}
