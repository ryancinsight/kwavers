//! Sparse Matrix Operations for Acoustic Simulations
//!
//! Provides efficient sparse matrix structures and operations for:
//! - Large-scale linear systems
//! - Reconstruction algorithms
//! - Iterative solvers
//!
//! # Architecture
//! - Modular design with clear separation of formats
//! - Zero-copy operations where possible
//! - Literature-based implementations
//!
//! # References
//! - Davis (2006): "Direct methods for sparse linear systems"
//! - Saad (2003): "Iterative methods for sparse linear systems"
//!
//! # Note
//!
//! Beamforming-specific sparse matrix operations have been migrated to
//! `analysis::signal_processing::beamforming::utils::sparse` to enforce
//! correct architectural layering.

pub mod coo;
pub mod csr;
pub mod eigenvalue;
pub mod solver;

pub use coo::CoordinateMatrix;
pub use csr::CompressedSparseRowMatrix;
pub use eigenvalue::EigenvalueSolver;
pub use solver::{IterativeSolver, SolverConfig};
