//! Sparse matrix types and operations.
//!
//! This module provides sparse matrix types used throughout the codebase for
//! boundary element and finite element solvers.
//!
//! # Architecture
//!
//! The canonical pipeline is *assemble in COO → to_csr → run CSR kernels*.
//! CSR is the solve/kernel target consumed by SpMV and SpMM operations.

// Re-export from leto-ops for compatibility
pub use leto_ops::application::sparse::{
    csc_spmv, csc_spmv_into, csr_to_dense, factor_numeric,
    factor_symbolic_with_ordering, sparse_lu_solve, spgemm, spmm, spmm_into, spmv, spmv_into,
    CooMatrix, CscColumn, CscMatrix, CsrMatrix, CsrRow, NumericLu, OrderingStrategy,
    OwnedNumericLu, SparseLuSolver, SymbolicLu, DENSE_LIMIT_DEFAULT,
};

// Re-export domain-specific types
pub use csr::CompressedSparseRowMatrix;
pub use solver::{IterativeSolver, SolverConfig, SparsePreconditioner};
pub use matfree::{MatFreeOperator, MatFreeResult, solve_lsqr_matfree};

// Re-export Complex64 for convenience
pub use eunomia::Complex64;

// Domain-specific sparse matrix type for BEM/FEM
pub mod csr;

// Iterative solver wrappers
pub mod solver;

// Matrix-free linear operators and LSQR solver
pub mod matfree;
