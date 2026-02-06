//! Iterative Solvers for Large-Scale Linear Systems
//!
//! This module provides iterative algorithms for solving linear systems,
//! particularly useful for large sparse matrices.
//!
//! **Available Solvers**:
//! - **LSQR**: Least-squares QR for overdetermined systems (Paige & Saunders 1982)
//! - **CG**: Conjugate Gradient (implemented in sparse module)
//! - **BiCGSTAB**: BiConjugate Gradient Stabilized (implemented in sparse module)
//!
//! **When to Use Iterative Methods**:
//! - Matrix is large and sparse (millions of unknowns)
//! - Direct methods too expensive in memory or time
//! - Overdetermined or least-squares problems
//! - Matrix-free applications (only matrix-vector products available)

pub mod lsqr;

pub use lsqr::{LsqrConfig, LsqrResult, LsqrSolver, StopReason};
