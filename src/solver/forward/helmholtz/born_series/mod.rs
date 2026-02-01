//! Born Series Methods for Helmholtz Equation
//!
//! This module implements Born series approximations for solving the acoustic
//! Helmholtz equation in heterogeneous media. The Born series provides a
//! perturbative expansion that is particularly effective for weakly scattering
//! media and serves as the foundation for more advanced iterative methods.
//!
//! ## Mathematical Foundation
//!
//! The Born approximation decomposes the total field ψ into incident and scattered components:
//! ```text
//! ψ = ψⁱ + ψˢ
//! ```
//!
//! Substituting into the Helmholtz equation ∇²ψ + k²(1 + V)ψ = 0 gives:
//! ```text
//! ∇²ψˢ + k²ψˢ = -k²V ψⁱ - k²V ψˢ
//! ```
//!
//! The first Born approximation neglects the Vψˢ term:
//! ```text
//! ∇²ψˢ + k²ψˢ = -k²V ψⁱ
//! ```
//!
//! ## Higher-Order Born Series
//!
//! The nth-order Born approximation includes scattering from previous orders:
//! ```text
//! ∇²ψₙ + k²ψₙ = -k²V ψ_{n-1}
//! ```
//!
//! Where ψₙ is the nth-order scattered field contribution.
//!
//! ## Convergent Born Series (CBS)
//!
//! The CBS method renormalizes the series to improve convergence:
//! ```text
//! ψₙ = ψ_{n-1} - G * (k²V ψ_{n-1})
//! ```
//!
//! Where G is the Green's function operator.
//!
//! ## References
//!
//! 1. Stanziola, A., et al. (2025). "Iterative Born Solver for the Acoustic
//!    Helmholtz Equation with Heterogeneous Sound Speed and Density"
//!
//! 2. Sun, Y., et al. (2025). "A viscoacoustic wave equation solver using
//!    modified Born series"

pub mod convergent;
pub mod iterative;
pub mod modified;
pub mod workspace;

// Explicit re-exports of Born series solver types
pub use convergent::ConvergentBornSolver;
pub use iterative::IterativeBornSolver;
pub use modified::ModifiedBornSolver;
pub use workspace::BornWorkspace;

/// Born series approximation order
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BornOrder {
    /// First-order Born approximation (single scattering)
    First,
    /// Second-order Born approximation
    Second,
    /// Third-order Born approximation
    Third,
    /// Higher-order with renormalization
    Convergent,
}

/// Configuration for Born series solvers
#[derive(Debug, Clone)]
pub struct BornConfig {
    /// Maximum order for Born series expansion
    pub max_order: BornOrder,
    /// Maximum number of iterations for convergent methods
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Enable renormalization for improved convergence
    pub enable_renormalization: bool,
    /// Use FFT for Green's function computation
    pub use_fft_green: bool,
    /// Include evanescent waves in Green's function
    pub include_evanescent: bool,
    /// Enable adaptive mesh refinement
    pub adaptive_meshing: bool,
    /// Maximum refinement levels for adaptive meshing
    pub max_refinement_levels: usize,
    /// Refinement threshold for adaptive meshing
    pub refinement_threshold: f64,
}

impl Default for BornConfig {
    fn default() -> Self {
        Self {
            max_order: BornOrder::Convergent,
            max_iterations: 50,
            tolerance: 1e-8,
            enable_renormalization: true,
            use_fft_green: true,
            include_evanescent: false,
            adaptive_meshing: false,
            max_refinement_levels: 3,
            refinement_threshold: 0.1,
        }
    }
}
