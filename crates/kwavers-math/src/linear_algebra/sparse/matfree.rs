//! Matrix-free linear operator interface and LSQR solver.
//!
//! Provides the `MatFreeOperator` trait and a `solve_lsqr_matfree` function
//! for solving linear systems `A x = b` where `A` is only accessible via
//! matrix-vector products (no explicit matrix storage).
//!
//! The solver delegates to [`leto_ops::LsqrSolver`] under the hood.

use leto::Array1;
use leto_ops::{LinearOperator, LsqrConfig, LsqrResult, LsqrSolver};

/// Matrix-free linear operator for iterative solvers.
///
/// This trait abstracts over linear operators that can be applied via
/// matrix-vector products without explicit matrix storage. Implementors
/// must provide both forward and transpose operations for use in
/// Krylov subspace methods like LSQR.
pub trait MatFreeOperator: Clone + Send + Sync {
    /// Number of rows (output dimension).
    fn rows(&self) -> usize;
    /// Number of columns (input dimension).
    fn cols(&self) -> usize;
    /// Matrix-vector product: `y = A * x`.
    fn matvec(&self, x: &[f64], y: &mut [f64]);
    /// Transpose matrix-vector product: `x = A^T * y`.
    fn t_matvec(&self, y: &[f64], x: &mut [f64]);
}

/// Result of `solve_lsqr_matfree` containing the solution and convergence history.
#[derive(Debug, Clone, PartialEq)]
pub struct MatFreeResult {
    /// The computed solution vector.
    pub solution: Vec<f64>,
    /// Per-iteration objective function values (residual norm history).
    pub objective_history: Vec<f64>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the solver converged.
    pub converged: bool,
}

impl MatFreeResult {
    /// Create a new LSQR result.
    #[must_use]
    pub fn new(solution: Vec<f64>, objective_history: Vec<f64>, iterations: usize, converged: bool) -> Self {
        Self {
            solution,
            objective_history,
            iterations,
            converged,
        }
    }
}

/// Adapter that bridges [`MatFreeOperator`] to [`LinearOperator<f64>`].
#[derive(Clone)]
pub struct MatFreeOperatorAdapter<Op: MatFreeOperator> {
    op: Op,
}

impl<Op: MatFreeOperator> MatFreeOperatorAdapter<Op> {
    #[must_use]
    pub fn new(op: Op) -> Self {
        Self { op }
    }
}

impl<Op: MatFreeOperator> LinearOperator<f64> for MatFreeOperatorAdapter<Op> {
    fn apply(&self, x: &Array1<f64>, y: &mut Array1<f64>) -> leto::Result<()> {
        let mut y_buf = vec![0.0_f64; y.len()];
        self.op.matvec(x.as_slice().unwrap(), &mut y_buf);
        y.as_slice_mut().unwrap().copy_from_slice(&y_buf);
        Ok(())
    }

    fn size(&self) -> usize {
        self.op.rows().max(self.op.cols())
    }

    fn nrows(&self) -> usize {
        self.op.rows()
    }

    fn ncols(&self) -> usize {
        self.op.cols()
    }

    fn apply_transpose(&self, x: &Array1<f64>, y: &mut Array1<f64>) -> leto::Result<()> {
        let mut y_buf = vec![0.0_f64; y.len()];
        self.op.t_matvec(x.as_slice().unwrap(), &mut y_buf);
        y.as_slice_mut().unwrap().copy_from_slice(&y_buf);
        Ok(())
    }
}

/// Solve `A x = b` using the LSQR algorithm with Tikhonov damping.
///
/// Delegates to [`LsqrSolver`] from `leto-ops`.
///
/// # Arguments
/// * `op` - Matrix-free operator implementing `MatFreeOperator`.
/// * `b` - Right-hand side vector.
/// * `config` - LSQR configuration including damping, tolerances, max iterations.
///
/// # Returns
/// * [`MatFreeResult`] containing the solution and convergence history.
pub fn solve_lsqr_matfree(
    op: &impl MatFreeOperator,
    b: &[f64],
    config: &LsqrConfig,
) -> MatFreeResult {
    let m = op.rows();
    let n = op.cols();

    if m == 0 || n == 0 || b.is_empty() {
        return MatFreeResult::new(
            vec![0.0; n],
            vec![],
            0,
            false,
        );
    }

    let solver = LsqrSolver::new(*config);
    let mut b_arr = Array1::zeros([m]);
    b_arr.as_slice_mut().unwrap().copy_from_slice(b);
    let adapter = MatFreeOperatorAdapter::new(op.clone());
    let result: LsqrResult = solver.solve(&adapter, &b_arr);

    MatFreeResult::new(
        result.solution.as_slice().unwrap().to_vec(),
        result.residual_history,
        result.iterations,
        result.converged,
    )
}

// LsqrConfig is imported from leto_ops and used directly in the public API.

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct DiagonalOperator {
        diag: f64,
    }

    impl MatFreeOperator for DiagonalOperator {
        fn rows(&self) -> usize {
            3
        }
        fn cols(&self) -> usize {
            3
        }
        fn matvec(&self, x: &[f64], y: &mut [f64]) {
            assert_eq!(x.len(), 3);
            assert_eq!(y.len(), 3);
            for i in 0..3 {
                y[i] = self.diag * x[i];
            }
        }
        fn t_matvec(&self, y: &[f64], x: &mut [f64]) {
            assert_eq!(y.len(), 3);
            assert_eq!(x.len(), 3);
            for i in 0..3 {
                x[i] = self.diag * y[i];
            }
        }
    }

    #[test]
    fn test_lsqr_solve_diagonal() {
        let op = DiagonalOperator { diag: 2.0 };
        let b = vec![4.0, 6.0, 8.0];
        let config = LsqrConfig {
            max_iterations: 1000,
            damping: 0.0,
            atol: 1e-6,
            btol: 1e-6,
            tolerance: 1e-4,
        };
        let result = solve_lsqr_matfree(&op, &b, &config);

        assert!(result.converged, "LSQR should converge for a diagonal system");
        assert!((result.solution[0] - 2.0).abs() < 1e-4);
        assert!((result.solution[1] - 3.0).abs() < 1e-4);
        assert!((result.solution[2] - 4.0).abs() < 1e-4);
    }

    #[test]
    fn test_lsqr_identity() {
        let op = DiagonalOperator { diag: 1.0 };
        let b = vec![1.0, 2.0, 3.0];
        let config = LsqrConfig {
            max_iterations: 5,
            damping: 0.0,
            atol: 1e-4,
            btol: 1e-4,
            tolerance: 1e-2,
        };
        let result = solve_lsqr_matfree(&op, &b, &config);

        eprintln!("LSQR identity: converged={}, iterations={}, solution={:?}",
                  result.converged, result.iterations, result.solution);

        assert!(result.converged, "LSQR should converge for identity system");
        assert!((result.solution[0] - 1.0).abs() < 1e-3);
        assert!((result.solution[1] - 2.0).abs() < 1e-3);
        assert!((result.solution[2] - 3.0).abs() < 1e-3);
    }
}
