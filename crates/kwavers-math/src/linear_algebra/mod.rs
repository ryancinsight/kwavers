//! Linear algebra operations.
//!
//! Submodules:
//! - `complex`: complex-matrix solve and inversion
//! - `eigendecomposition`: general eigendecomposition
//! - `iterative`: iterative solvers (LSQR)
//! - `norms`: vector norms
//! - `numeric_ops`: generic float trait (`NumericOps`)
//! - `ext`: fluent ndarray extension trait (`LinearAlgebraExt`) and `norm_l2`
//! - `tolerance`: numerical tolerance constants
//! - `sparse`: sparse matrix operations (CSR, COO)
//!
//! Note: Basic linear algebra operations (solve, inv, LU, QR, Cholesky, symmetric eigen)
//! are provided by leto-ops and should be used directly instead of the deprecated
//! `basic` and `eigen` modules.

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

#[cfg(test)]
mod tests;
