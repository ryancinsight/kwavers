//! Linear algebra operations.
//!
//! Submodules:
//! - `basic`: real-matrix solve, inversion, decomposition
//! - `complex`: complex-matrix solve and inversion
//! - `eigen`: eigendecomposition (real symmetric, complex Hermitian)
//! - `norms`: vector norms
//! - `numeric_ops`: generic float trait (`NumericOps`)
//! - `ext`: fluent ndarray extension trait (`LinearAlgebraExt`) and `norm_l2`
//! - `tolerance`: numerical tolerance constants

pub mod basic;
pub mod complex;
pub mod eigen;
pub mod eigendecomposition;
pub mod ext;
pub mod iterative;
pub mod norms;
pub mod numeric_ops;
pub mod sparse;
pub mod tolerance;

pub use basic::LinearAlgebra;
pub use complex::ComplexLinearAlgebra;
pub use eigen::EigenDecomposition;
pub use eigendecomposition::{EigenResult, EigenSolver, EigenSolverConfig};
pub use ext::{norm_l2, LinearAlgebraExt};
pub use norms::VectorOperations;
pub use numeric_ops::NumericOps;

#[cfg(test)]
mod tests;
