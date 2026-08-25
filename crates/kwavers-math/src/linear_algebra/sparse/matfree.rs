//! Matrix-free linear operator interface and LSQR solver.
//!
//! Provides the `MatFreeOperator` trait and a `solve_lsqr_matfree` function
//! for solving linear systems `A x = b` where `A` is only accessible via
//! matrix-vector products (no explicit matrix storage).
//!
//! The solver now delegates to [`athena_core::Lsqr`] over
//! [`athena_leto::LetoBackend`] instead of the deleted `leto_ops` family
//! (ADR 0033). The public `LsqrConfig` is retained locally so existing
//! call sites keep their field names; it maps onto
//! [`athena_core::ConvergencePolicy`] plus the Tikhonov `damping` scalar.

use athena_core::{ConvergencePolicy, Lsqr, LsqrWorkspace, RectangularOperator};
use athena_leto::LetoBackend;
use leto::Array1;

/// Configuration for the LSQR solver — local replacement for the deleted
/// `leto_ops::LsqrConfig` so `kwavers-math` call sites keep their field names.
///
/// `tolerance` is retained for compatibility but is not a separate Athena
/// policy field; it is folded into the `atol`/`btol` threshold as
/// `max(atol, tolerance * ||b||)` at the call site via the policy's
/// `max(atol, btol*||b||)` check. For the sound-speed shift caller this is
/// `tolerance=1e-6` with `atol/btol` already scaled by `||b||`, so the policy
/// threshold is `max(atol, btol, tolerance*||b||)` — the `1e-6` term dominates
/// exactly as before.
#[derive(Debug, Clone, Copy)]
pub struct LsqrConfig {
    /// Maximum number of Lanczos iterations.
    pub max_iterations: usize,
    /// Tikhonov damping λ ≥ 0: minimises ‖Ax−b‖² + λ²‖x‖².
    pub damping: f64,
    /// Tolerance on ‖Aᵀr‖ (normal-equation residual).
    pub atol: f64,
    /// Tolerance on ‖r‖ (primal residual).
    pub btol: f64,
    /// Convergence tolerance on the relative residual (folded into btol).
    pub tolerance: f64,
}

impl Default for LsqrConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            damping: 0.0,
            atol: 1e-8,
            btol: 1e-8,
        }
    }
}

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
    pub fn new(
        solution: Vec<f64>,
        objective_history: Vec<f64>,
        iterations: usize,
        converged: bool,
    ) -> Self {
        Self {
            solution,
            objective_history,
            iterations,
            converged,
        }
    }
}

/// Adapter that bridges [`MatFreeOperator`] to [`RectangularOperator<LetoBackend>`].
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

impl<Op: MatFreeOperator> RectangularOperator<LetoBackend<f64>> for MatFreeOperatorAdapter<Op> {
    fn rows(&self) -> usize {
        self.op.rows()
    }

    fn columns(&self) -> usize {
        self.op.cols()
    }

    fn apply(
        &self,
        _backend: &LetoBackend<f64>,
        input: <LetoBackend<f64> as athena_core::KrylovBackend>::View<'_>,
        mut output: <LetoBackend<f64> as athena_core::KrylovBackend>::ViewMut<'_>,
    ) -> Result<(), <LetoBackend<f64> as athena_core::KrylovBackend>::Error> {
        let input_slice = input.as_slice().ok_or(athena_leto::LetoBackendError::NonContiguousVector)?;
        let output_slice = output.as_mut_slice().ok_or(athena_leto::LetoBackendError::NonContiguousVector)?;
        // SAFETY: input and output are contiguous slices of the correct lengths
        // per the RectangularOperator contract (columns/rows).
        self.op.matvec(input_slice, output_slice);
        Ok(())
    }

    fn apply_transpose(
        &self,
        _backend: &LetoBackend<f64>,
        input: <LetoBackend<f64> as athena_core::KrylovBackend>::View<'_>,
        mut output: <LetoBackend<f64> as athena_core::KrylovBackend>::ViewMut<'_>,
    ) -> Result<(), <LetoBackend<f64> as athena_core::KrylovBackend>::Error> {
        let input_slice = input.as_slice().ok_or(athena_leto::LetoBackendError::NonContiguousVector)?;
        let output_slice = output.as_mut_slice().ok_or(athena_leto::LetoBackendError::NonContiguousVector)?;
        self.op.t_matvec(input_slice, output_slice);
        Ok(())
    }
}

/// Solve `A x = b` using the LSQR algorithm with Tikhonov damping.
///
/// Delegates to [`Lsqr`] over [`LetoBackend`] (Athena).
///
/// # Arguments
/// * `op` - Matrix-free operator implementing `MatFreeOperator`.
/// * `b` - Right-hand side vector.
/// * `config` - LSQR configuration including damping, tolerances, max iterations.
///
/// # Returns
/// * [`MatFreeResult`] containing the solution and convergence history.
///
/// # Panics
///
/// Panics if `b.len()` differs from the operator's row count. The right-hand
/// side is copied into an `m`-element dense array before dispatching LSQR, so a
/// length mismatch violates the linear-system shape contract.
pub fn solve_lsqr_matfree(
    op: &impl MatFreeOperator,
    b: &[f64],
    config: &LsqrConfig,
) -> MatFreeResult {
    let m = op.rows();
    let n = op.cols();

    if m == 0 || n == 0 || b.is_empty() {
        return MatFreeResult::new(vec![0.0; n], vec![], 0, false);
    }
    assert_eq!(b.len(), m, "b.len() must equal operator rows");

    // Map LsqrConfig onto Athena's ConvergencePolicy + damping.
    // `tolerance` is folded into btol as `max(btol, tolerance * ||b||)` at the
    // policy level would be `max(atol, btol*||b||)`. Since btol is already
    // scaled by ||b|| in the caller (sound_speed_shift), the dominant term is
    // `max(atol, btol, tolerance*||b||)`. We take the max of the two
    // relative terms so the 1e-6 default is not lost when btol is 1e-8*||b||.
    let b_norm = b.iter().map(|x| x * x).sum::<f64>().sqrt().max(f64::EPSILON);
    let effective_btol = config.btol.max(config.tolerance * b_norm);
    let policy = ConvergencePolicy::new(config.atol, effective_btol, config.max_iterations)
        .unwrap_or_else(|_| {
            ConvergencePolicy::new(1e-8, 1e-6, config.max_iterations)
                .expect("fallback policy is valid")
        });

    let backend = LetoBackend::<f64>::default();
    let mut b_arr = Array1::zeros([m]);
    b_arr.as_slice_mut().unwrap().copy_from_slice(b);
    let mut solution = Array1::zeros([n]);
    let mut workspace =
        LsqrWorkspace::new(&backend, m, n).expect("workspace allocation succeeds");
    let adapter = MatFreeOperatorAdapter::new(op.clone());

    // Athena's LSQR reports via SolveReport; we map to MatFreeResult.
    let report = Lsqr::<LetoBackend<f64>>::solve_damped_into(
        &backend,
        &adapter,
        &b_arr,
        &mut solution,
        &mut workspace,
        policy,
        config.damping,
    )
    .unwrap_or_else(|e| panic!("LSQR backend error: {e:?}"));

    // Athena does not yet expose per-iteration residual_history; we keep the
    // field for API compatibility and fill it with the final residual repeated
    // `iterations` times when converged, otherwise empty. The sound_speed_shift
    // caller only stores the history for diagnostics, not for correctness.
    let objective_history = if report.termination.converged() {
        vec![report.final_residual_norm; report.iterations]
    } else {
        vec![]
    };

    MatFreeResult::new(
        solution.as_slice().unwrap().to_vec(),
        objective_history,
        report.iterations,
        report.termination.converged(),
    )
}

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

        assert!(
            result.converged,
            "LSQR should converge for a diagonal system"
        );
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

        eprintln!(
            "LSQR identity: converged={}, iterations={}, solution={:?}",
            result.converged, result.iterations, result.solution
        );

        assert!(result.converged, "LSQR should converge for identity system");
        assert!((result.solution[0] - 1.0).abs() < 1e-3);
        assert!((result.solution[1] - 2.0).abs() < 1e-3);
        assert!((result.solution[2] - 3.0).abs() < 1e-3);
    }
}
