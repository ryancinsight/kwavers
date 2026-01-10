//! Helmholtz Equation Solvers
//!
//! This module provides advanced solvers for the acoustic Helmholtz equation,
//! including Born series methods for heterogeneous media and iterative approaches
//! for strong scattering scenarios.
//!
//! ## Mathematical Foundation
//!
//! The acoustic Helmholtz equation for heterogeneous media:
//! ```text
//! ∇²ψ + k²(1 + V)ψ = S
//! ```
//!
//! Where:
//! - ψ: acoustic field (pressure or velocity potential)
//! - k: wavenumber (ω/c₀)
//! - V: heterogeneity potential (related to density/sound speed variations)
//! - S: source terms
//!
//! ## Born Series Methods
//!
//! The Born series provides a perturbative solution for the scattered field:
//! ```text
//! ψ = ψ₀ + ψ₁ + ψ₂ + ... + ψₙ + ...
//! ```
//!
//! Where each term satisfies:
//! ```text
//! ∇²ψₙ + k²ψₙ = -k²V ψ_{n-1}
//! ```
//!
//! ## Key Features
//!
//! - **Convergent Born Series**: Renormalized series for improved convergence
//! - **Iterative Born**: Fixed-point iteration for strong scattering
//! - **Modified Born**: Adapted for viscoacoustic media
//! - **Matrix-free FFT implementation**: Efficient for large-scale problems
//! - **Heterogeneous density support**: Full acoustic parameter variations

pub mod born_series;
pub mod preconditioners;

/// Configuration for Helmholtz solvers
#[derive(Debug, Clone)]
pub struct HelmholtzConfig {
    /// Maximum number of iterations for iterative methods
    pub max_iterations: usize,
    /// Convergence tolerance for iterative solvers
    pub tolerance: f64,
    /// Enable renormalization for convergent Born series
    pub enable_renormalization: bool,
    /// FFT-based matrix-free operations
    pub use_fft: bool,
    /// Preconditioner type for iterative methods
    pub preconditioner: PreconditionerType,
}

impl Default for HelmholtzConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            enable_renormalization: true,
            use_fft: true,
            preconditioner: PreconditionerType::None,
        }
    }
}

/// Preconditioner options for iterative Helmholtz solvers
#[derive(Debug, Clone, Copy)]
pub enum PreconditionerType {
    /// No preconditioning
    None,
    /// Diagonal preconditioning
    Diagonal,
    /// Incomplete LU factorization
    ILU,
    /// Multigrid preconditioning
    Multigrid,
}

/// Preconditioner trait for Helmholtz solvers
pub trait Preconditioner {
    /// Apply preconditioner to a field
    fn apply(
        &self,
        input: &ndarray::ArrayView3<num_complex::Complex64>,
        output: &mut ndarray::ArrayViewMut3<num_complex::Complex64>,
    ) -> crate::core::error::KwaversResult<()>;

    /// Setup preconditioner for given wavenumber and medium
    fn setup(
        &mut self,
        wavenumber: f64,
        medium: &dyn crate::domain::medium::Medium,
        grid: &crate::domain::grid::Grid,
    ) -> crate::core::error::KwaversResult<()>;
}

pub use born_series::*;
