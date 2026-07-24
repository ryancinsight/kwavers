//! Iterative solvers - SSOT: leto_ops application linalg iterative.
//!
//! Re-exported here as the kwavers vocabulary so higher layers depend on one
//! import path while the implementation lives in leto-ops.

pub use leto_ops::{
    BiCGSTAB, ConjugateGradient, GMRES, LsqrConfig, LsqrResult, LsqrSolver, LsqrStopReason,
    LinearOperator, LinearSolver, Preconditioner, IterativeLinearSolver, IterativeSolverConfig,
    IdentityPreconditioner, JacobiPreconditioner, ILUPreconditioner,
};

/// LSQR solver wrapper preserving the kwavers import path.
pub mod lsqr {
    pub use leto_ops::{LsqrConfig, LsqrResult, LsqrSolver, LsqrStopReason};
}
