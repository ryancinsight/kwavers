//! LSQR iterative solver for sparse least-squares problems.
//!
//! Implements LSQR (Paige & Saunders 1982) via Lanczos bidiagonalisation
//! and Givens QR factorisation.

pub mod matfree;
pub mod solver;
#[cfg(test)]
mod tests;
pub mod types;

pub use matfree::{solve_lsqr_matfree, MatFreeOperator, MatFreeResult};
pub use solver::LsqrSolver;
pub use types::{LsqrConfig, LsqrResult, StopReason};
