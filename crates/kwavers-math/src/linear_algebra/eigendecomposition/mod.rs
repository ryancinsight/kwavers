//! Eigenvalue decomposition for complex Hermitian matrices.
//!
//! SSOT: leto_ops application linalg eigen and hermitian.
//!
//! This module provides a kwavers-vocabulary wrapper around leto-ops
//! eigensolvers, preserving the existing API surface while delegating
//! to the SSOT implementation.

use eunomia::Complex64;
use kwavers_core::error::KwaversResult;
use leto::{Array1, Array2};

/// Eigenvalue decomposition solver - delegates to leto-ops.
#[derive(Debug)]
pub struct EigenSolver;

/// Configuration for eigenvalue solver (maps to leto_ops HermitianEigenConfig).
#[derive(Debug, Clone, Copy)]
pub struct EigenSolverConfig {
    /// Convergence tolerance (default: 1e-10)
    pub tolerance: f64,
    /// Maximum number of iterations (default: 1000)
    pub max_iterations: usize,
    /// Whether to sort eigenvalues in descending order (default: true)
    pub sort_descending: bool,
    /// Estimate condition number (default: true)
    pub estimate_condition: bool,
}

impl Default for EigenSolverConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-10,
            max_iterations: 1000,
            sort_descending: true,
            estimate_condition: true,
        }
    }
}

impl EigenSolverConfig {
    fn to_leto_config(self) -> leto_ops::HermitianEigenConfig {
        leto_ops::HermitianEigenConfig {
            tolerance: self.tolerance,
            max_iterations: self.max_iterations,
            sort_descending: self.sort_descending,
            estimate_condition: self.estimate_condition,
        }
    }
}

/// Result of eigenvalue decomposition with diagnostic information.
#[derive(Debug, Clone)]
pub struct EigenResult {
    /// Eigenvalues (sorted if config.sort_descending = true)
    pub eigenvalues: Array1<f64>,
    /// Eigenvectors as columns (corresponding to eigenvalues)
    pub eigenvectors: Array2<Complex64>,
    /// Number of iterations used
    pub iterations: usize,
    /// Final off-diagonal norm (convergence criterion)
    pub off_diagonal_norm: f64,
    /// Condition number estimate kappa(A) = lambda_max / lambda_min
    pub condition_number: Option<f64>,
    /// Algorithm used
    pub algorithm: String,
}

impl EigenSolver {
    /// QR algorithm with Wilkinson shift for complex Hermitian matrices.
    /// Delegates to leto_ops::hermitian_eigen_qr.
    pub fn qr_algorithm(
        matrix: &Array2<Complex64>,
        config: EigenSolverConfig,
    ) -> KwaversResult<EigenResult> {
        let leto_config = config.to_leto_config();
        let result = leto_ops::hermitian_eigen_qr(matrix, leto_config)?;
        Ok(EigenResult {
            eigenvalues: result.eigenvalues,
            eigenvectors: result.eigenvectors,
            iterations: result.iterations,
            off_diagonal_norm: result.off_diagonal_norm,
            condition_number: result.condition_number,
            algorithm: "QR with Wilkinson shift (leto-ops)".to_string(),
        })
    }

    /// Jacobi method for complex Hermitian matrices.
    /// Delegates to leto_ops::hermitian_eigen_jacobi.
    pub fn jacobi_hermitian(
        matrix: &Array2<Complex64>,
        config: EigenSolverConfig,
    ) -> KwaversResult<EigenResult> {
        let leto_config = config.to_leto_config();
        let result = leto_ops::hermitian_eigen_jacobi(matrix, leto_config)?;
        Ok(EigenResult {
            eigenvalues: result.eigenvalues,
            eigenvectors: result.eigenvectors,
            iterations: result.iterations,
            off_diagonal_norm: result.off_diagonal_norm,
            condition_number: result.condition_number,
            algorithm: "Jacobi (leto-ops)".to_string(),
        })
    }
}
