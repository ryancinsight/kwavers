//! Sparse Matrix Operations for Acoustic Simulations
//!
//! Provides efficient sparse matrix structures and operations for:
//! - Beamforming computations
//! - Large-scale linear systems
//! - Reconstruction algorithms
//!
//! # Architecture
//! - Modular design with clear separation of formats
//! - Zero-copy operations where possible
//! - Literature-based implementations
//!
//! # References
//! - Davis (2006): "Direct methods for sparse linear systems"
//! - Saad (2003): "Iterative methods for sparse linear systems"

pub mod beamforming;
pub mod coo;
pub mod csr;
pub mod eigenvalue;
pub mod solver;

pub use beamforming::BeamformingMatrix;
pub use coo::CoordinateMatrix;
pub use csr::CompressedSparseRowMatrix;
pub use eigenvalue::EigenvalueSolver;
pub use solver::{IterativeSolver, SolverConfig};
