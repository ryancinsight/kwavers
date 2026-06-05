//! Advanced Eigenvalue Decomposition for Complex Hermitian Matrices
//!
//! This module implements state-of-the-art eigenvalue algorithms optimized for:
//! - Complex Hermitian matrices (covariance matrices in beamforming)
//! - Signal processing applications (MUSIC, ESPRIT)
//! - FDA compliance validation
//!
//! ## Algorithms Implemented
//!
//! ### QR Algorithm with Wilkinson Shift
//! - Iterative eigenvalue computation with implicit QR iterations
//! - Wilkinson shift for improved convergence
//! - Suitable for dense matrices up to ~1000×1000
//! - O(n³) complexity but with good constant factors
//!
//! ### Jacobi Method for Hermitian Matrices
//! - Jacobi eigenvalue algorithm for complex Hermitian matrices
//! - Guaranteed convergence with high numerical stability
//! - O(n³) but often faster for small matrices (n < 100)
//! - Used as fallback for ill-conditioned problems
//!
//! ## Theoretical Foundations
//!
//! **Schur Decomposition**: A = Q·T·Q^H where Q is unitary, T is upper triangular
//!
//! **Rayleigh Quotient**: R(x) = x^H·A·x / (x^H·x)
//!
//! **Condition Number**: κ(A) = λ_max / λ_min
//!
//! ## References
//!
//! - Golub & Van Loan (2013): "Matrix Computations" (4th ed)
//! - Parlett (1998): "The Symmetric Eigenvalue Problem"
//! - Wilkinson (1965): "The Algebraic Eigenvalue Problem"

mod algorithms;
mod helpers;
#[cfg(test)]
mod tests;

use ndarray::{Array1, Array2};
use num_complex::Complex;

/// Advanced eigenvalue decomposition with multiple algorithms
#[derive(Debug)]
pub struct EigenSolver;

/// Configuration for eigenvalue solver
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

/// Result of eigenvalue decomposition with diagnostic information
#[derive(Debug, Clone)]
pub struct EigenResult {
    /// Eigenvalues (sorted if config.sort_descending = true)
    pub eigenvalues: Array1<f64>,
    /// Eigenvectors as columns (corresponding to eigenvalues)
    pub eigenvectors: Array2<Complex<f64>>,
    /// Number of iterations used
    pub iterations: usize,
    /// Final off-diagonal norm (convergence criterion)
    pub off_diagonal_norm: f64,
    /// Condition number estimate κ(A) = λ_max / λ_min
    pub condition_number: Option<f64>,
    /// Algorithm used
    pub algorithm: String,
}
