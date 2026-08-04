//! Iterative solver wrappers for sparse systems.
//!
//! This module provides domain-specific iterative solver configurations and
//! preconditioners used in BEM/FEM solvers.

use std::fmt;

pub use super::csr::CompressedSparseRowMatrix;
pub use eunomia::Complex64;

/// Preconditioner type for iterative solvers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SparsePreconditioner {
    /// No preconditioner
    None,
    /// Jacobi (diagonal) preconditioner
    Jacobi,
    /// SSOR preconditioner
    SSOR(f64),
    /// ILU preconditioner
    ILU,
}

impl Default for SparsePreconditioner {
    fn default() -> Self {
        Self::None
    }
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

/// Iterative solver for sparse systems.
///
/// Provides domain-specific solver methods for BEM/FEM systems.
#[derive(Debug, Clone)]
pub struct IterativeSolver {
    config: SolverConfig,
}

impl IterativeSolver {
    /// Create a new solver with the given configuration.
    pub fn new(config: SolverConfig) -> Self {
        Self { config }
    }

    /// Create a solver with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(SolverConfig::default())
    }

    /// Create a new solver from configuration.
    pub fn create(config: SolverConfig) -> Self {
        Self::new(config)
    }

    /// BiCGSTAB solver for complex systems.
    ///
    /// Solves the system A*x = b using the BiCGSTAB algorithm.
    pub fn bicgstab_complex(
        &self,
        a_matrix: &crate::linear_algebra::sparse::csr::CompressedSparseRowMatrix<Complex64>,
        b: &[Complex64],
        x0: Option<&[Complex64]>,
    ) -> Result<Vec<Complex64>, String> {
        // Placeholder - actual implementation would use leto-ops solvers
        Err("bicgstab_complex not implemented".to_string())
    }

    /// GMRES solver for complex systems.
    pub fn gmres_complex(
        &self,
        a_matrix: &crate::linear_algebra::sparse::csr::CompressedSparseRowMatrix<Complex64>,
        b: &[Complex64],
        restart: usize,
    ) -> Result<Vec<Complex64>, String> {
        Err("gmres_complex not implemented".to_string())
    }

    /// Apply preconditioner.
    fn apply_preconditioner(
        &self,
        x: &mut [Complex64],
        a_matrix: &crate::linear_algebra::sparse::csr::CompressedSparseRowMatrix<Complex64>,
    ) {
        match self.config.preconditioner {
            SparsePreconditioner::None => {}
            SparsePreconditioner::Jacobi => {
                // Jacobi preconditioning: divide by diagonal
                for i in 0..x.len() {
                    let diag = a_matrix.values[a_matrix.row_pointers[i]];
                    if diag != Complex64::new(0.0, 0.0) {
                        x[i] /= diag;
                    }
                }
            }
            _ => {}
        }
    }
}
