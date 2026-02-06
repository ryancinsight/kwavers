//! Acoustic solver backend implementations
//!
//! This module provides concrete acoustic solver backends that adapt low-level
//! solvers to the simulation-facing `AcousticSolverBackend` trait interface.
//!
//! # Architecture
//!
//! ```text
//! simulation::AcousticWaveSolver
//!     ↓ uses
//! simulation::backends::AcousticSolverBackend (trait)
//!     ↑ implemented by
//! FdtdBackend (this module)
//!     ↓ wraps
//! solver::forward::fdtd::FdtdSolver
//! ```
//!
//! # Design Rationale
//!
//! By placing backends here:
//! - **Clinical layer** never knows about solver details
//! - **Simulation layer** orchestrates backends without exposing them to clinical
//! - **Clear separation**: Clinical → Simulation → Solver
//! - **Testability**: Backends can be tested independently of clinical code

pub mod backend;
pub mod fdtd;

pub use backend::AcousticSolverBackend;
pub use fdtd::FdtdBackend;
