//! Iterative solver wrappers for sparse systems.
//!
//! This module provides domain-specific iterative solver configurations and
//! preconditioners used in BEM/FEM solvers.

use std::fmt;

pub use super::csr::CompressedSparseRowMatrix;
pub use eunomia::Complex64;

/// Preconditioner type for iterative solvers.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum SparsePreconditioner {
    /// No preconditioner
    #[default]
    None,
    /// Jacobi (diagonal) preconditioner
    Jacobi,
    /// SSOR preconditioner
    SSOR(f64),
    /// ILU preconditioner
    ILU,
}

impl fmt::Display for SparsePreconditioner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::Jacobi => write!(f, "Jacobi"),
            Self::SSOR(_) => write!(f, "SSOR"),
            Self::ILU => write!(f, "ILU"),
        }
    }
}

/// Configuration for iterative solvers.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub preconditioner: SparsePreconditioner,
    pub verbose: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            preconditioner: SparsePreconditioner::None,
            verbose: false,
        }
    }
}

/// Iterative solver configuration marker.
///
/// Placeholder for future preconditioner/config support.
/// Actual iterative solvers are accessed via `leto_ops::application::linalg`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IterativeSolver;

impl IterativeSolver {
    /// Create a new solver with the given configuration.
    #[must_use]
    pub fn new(_config: SolverConfig) -> Self {
        Self
    }

    /// Create a solver with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(SolverConfig::default())
    }

    /// Create a new solver from configuration.
    #[must_use]
    pub fn create(_config: SolverConfig) -> Self {
        Self::new(SolverConfig::default())
    }

    /// BiCGSTAB solver for complex systems (stub — use `leto_ops::application::linalg` for real solvers).
    pub fn bicgstab_complex(
        &self,
        _a_matrix: &CompressedSparseRowMatrix<Complex64>,
        _b: &[Complex64],
        _x0: Option<&[Complex64]>,
    ) -> Result<Vec<Complex64>, String> {
        Err("bicgstab_complex: use leto_ops::application::linalg for iterative solvers".to_string())
    }

    /// GMRES solver for complex systems (stub — use `leto_ops::application::linalg` for real solvers).
    pub fn gmres_complex(
        &self,
        _a_matrix: &CompressedSparseRowMatrix<Complex64>,
        _b: &[Complex64],
        _restart: usize,
    ) -> Result<Vec<Complex64>, String> {
        Err("gmres_complex: use leto_ops::application::linalg for iterative solvers".to_string())
    }
}
