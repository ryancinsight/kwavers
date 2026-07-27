//! Linear algebra operations.
//!
//! SSOT: leto-ops application linalg.
//!
//! All submodules delegate to leto-ops, providing a kwavers-vocabulary
//! wrapper that preserves the KwaversError contract.
//!
//! Submodules:
//! - eigendecomposition: Hermitian eigendecomposition (delegates to leto_ops::hermitian_eigen_*)
//! - iterative: iterative solvers (delegates to leto_ops::iterative)
//! - norms: vector norms (delegates to leto_ops::norm_*)
//! - sparse: sparse matrix operations (delegates to leto::CsrMatrix and leto_ops::iterative)
//! - tolerance: numerical tolerance constants (unique to kwavers)

pub mod eigendecomposition;
pub mod iterative;
pub mod norms;
pub mod sparse;
pub mod tolerance;

pub use eigendecomposition::{EigenResult, EigenSolver, EigenSolverConfig};
pub use norms::VectorOperations;
