//! Inverse Problem Solvers and Utilities
//!
//! This module provides mathematical foundations for solving inverse problems,
//! including regularization methods, iterative solvers, and algorithmic tools.
//!
//! **Structure**:
//! - `regularization`: Tikhonov, TV, L1, smoothness regularization (SSOT)
//! - `iterative`: CG, LSQR, BiCGSTAB solvers (via linear_algebra SSOT)
//!
//! **Design Principle**: Single Source of Truth (SSOT) for all inverse problem utilities.
//! All higher-level inverse solvers (SIRT, FWI, etc.) use these foundational components.

pub mod regularization;

pub use regularization::{
    ModelRegularizer1D, ModelRegularizer2D, ModelRegularizer3D, RegularizationConfig,
};
