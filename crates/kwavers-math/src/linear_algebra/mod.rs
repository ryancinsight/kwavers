//! Linear algebra operations.
//!
//! SSOT: leto-ops application linalg.
//!
//! All submodules delegate to leto-ops, providing a kwavers-vocabulary
//! wrapper that preserves the KwaversError contract.
//!
//! Submodules:
//! - complex: complex-matrix solve and inversion (delegates to leto_ops::complex_solve/complex_inv)
//! - eigendecomposition: Hermitian eigendecomposition (delegates to leto_ops::hermitian_eigen_*)
//! - ext: fluent leto::Array extension trait (delegates to leto_ops::solve/inv/eig)
//! - iterative: iterative solvers (delegates to leto_ops::iterative)
//! - norms: vector norms (delegates to leto_ops::norm_*)
//! - numeric_ops: generic float trait (delegates to leto element-wise ops)
//! - sparse: sparse matrix operations (delegates to leto::CsrMatrix and leto_ops::iterative)
//! - tolerance: numerical tolerance constants (unique to kwavers)

pub mod complex;
pub mod eigendecomposition;
pub mod ext;
pub mod iterative;
pub mod norms;
pub mod numeric_ops;
pub mod sparse;
pub mod tolerance;

pub use complex::ComplexLinearAlgebra;
pub use eigendecomposition::{EigenResult, EigenSolver, EigenSolverConfig};
pub use ext::{norm_l2, LinearAlgebraExt};
pub use norms::VectorOperations;
pub use numeric_ops::NumericOps;
