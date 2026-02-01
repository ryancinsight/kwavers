//! Preconditioners for Helmholtz equation solvers
//!
//! This module provides various preconditioning techniques to improve
//! convergence of iterative Helmholtz solvers, particularly for Born series methods.

pub mod diagonal;
pub mod sor;

// Explicit re-exports of preconditioner types
pub use diagonal::DiagonalPreconditioner;
// TODO: Implement SORPreconditioner
// pub use sor::SORPreconditioner;
