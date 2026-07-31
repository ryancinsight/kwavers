//! Iterative solvers - SSOT: leto_ops application linalg iterative.
//!
//! Re-exported here as the kwavers vocabulary so higher layers depend on one
//! import path while the implementation lives in leto-ops.

pub use leto_ops::{
    BiCGSTAB, ConjugateGradient, ILUPreconditioner, IdentityPreconditioner, IterativeLinearSolver,
    IterativeSolverConfig, JacobiPreconditioner, LinearOperator, LinearSolver, LsqrConfig,
    LsqrResult, LsqrSolver, LsqrStopReason, Preconditioner, GMRES,
};

/// LSQR solver wrapper preserving the kwavers import path.
pub mod lsqr {
    pub use leto_ops::{LsqrConfig, LsqrResult, LsqrSolver, LsqrStopReason};

    /// Type-alias sub-module so callers can write `lsqr::types::LsqrConfig`.
    pub mod types {
        pub use leto_ops::LsqrConfig;
    }

    /// Matrix-free LSQR surface: [`MatFreeOperator`] trait, [`MatFreeResult`],
    /// and the [`solve_lsqr_matfree`] driver.
    pub mod matfree {
        use super::{LsqrConfig, LsqrSolver};
        use leto::Array1;
        use leto_ops::LinearOperator;

        /// Slice-based matrix-free operator for the LSQR driver.
        pub trait MatFreeOperator: Send + Sync {
            /// Number of rows (output dimension).
            fn rows(&self) -> usize;
            /// Number of columns (input dimension).
            fn cols(&self) -> usize;
            /// Forward apply: `y ← A·x`.
            fn matvec(&self, x: &[f64], y: &mut [f64]);
            /// Transpose apply: `x ← Aᵀ·y`.
            fn t_matvec(&self, y: &[f64], x: &mut [f64]);
        }

        /// Result returned by [`solve_lsqr_matfree`].
        #[derive(Debug, Clone)]
        pub struct MatFreeResult {
            /// Solution vector.
            pub solution: Vec<f64>,
            /// Per-iteration residual norms (objective history).
            pub objective_history: Vec<f64>,
            /// Number of iterations performed.
            pub iterations: usize,
            /// Whether a convergence criterion was satisfied.
            pub converged: bool,
        }

        struct MatFreeAdapter<'a>(&'a dyn MatFreeOperator);

        impl LinearOperator<f64> for MatFreeAdapter<'_> {
            fn apply(&self, x: &Array1<f64>, y: &mut Array1<f64>) -> leto::Result<()> {
                let xs = x.as_slice().expect("contiguous Array1");
                let ys = y.as_slice_mut().expect("contiguous Array1");
                self.0.matvec(xs, ys);
                Ok(())
            }
            fn size(&self) -> usize {
                self.0.rows()
            }
            fn nrows(&self) -> usize {
                self.0.rows()
            }
            fn ncols(&self) -> usize {
                self.0.cols()
            }
            fn apply_transpose(&self, x: &Array1<f64>, y: &mut Array1<f64>) -> leto::Result<()> {
                let xs = x.as_slice().expect("contiguous Array1");
                let ys = y.as_slice_mut().expect("contiguous Array1");
                self.0.t_matvec(xs, ys);
                Ok(())
            }
        }

        /// Solve the least-squares problem `min ‖A·x − b‖` using LSQR.
        #[must_use]
        pub fn solve_lsqr_matfree(
            op: &dyn MatFreeOperator,
            b: &[f64],
            config: &LsqrConfig,
        ) -> MatFreeResult {
            let solver = LsqrSolver::new(*config);
            let b_array = Array1::from_vec(b.len(), b.to_vec()).expect("b slice fits Array1 shape");
            let adapter = MatFreeAdapter(op);
            let result = solver.solve(&adapter, &b_array);
            MatFreeResult {
                solution: result.solution.into_vec(),
                objective_history: result.residual_history,
                iterations: result.iterations,
                converged: result.converged,
            }
        }
    }

    pub use matfree::{solve_lsqr_matfree, MatFreeOperator, MatFreeResult};
}
