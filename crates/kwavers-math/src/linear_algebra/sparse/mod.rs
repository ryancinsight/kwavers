//! Sparse matrix operations - SSOT: leto_ops CsrMatrix and leto_ops iterative.
//!
//! Re-exported here as the kwavers vocabulary so higher layers depend on one
//! import path while the implementation lives in leto-ops.

pub mod csr;
pub mod solver;

/// Re-export for backward-compatible import path.
pub use csr::CompressedSparseRowMatrix;
