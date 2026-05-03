//! Configuration types for SIRT/ART/OSEM reconstruction.

use crate::math::inverse_problems::RegularizationConfig;
use ndarray::Array3;
use std::fmt;

/// Algorithm selection for iterative reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SirtAlgorithm {
    /// Simultaneous Iterative Reconstruction Technique (stable, slow).
    Sirt,
    /// Algebraic Reconstruction Technique (faster, cyclic).
    Art,
    /// Ordered Subset Expectation Maximization (practical fast convergence).
    Osem { num_subsets: usize },
}

impl fmt::Display for SirtAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Sirt => write!(f, "SIRT"),
            Self::Art => write!(f, "ART"),
            Self::Osem { num_subsets } => write!(f, "OSEM (subsets={})", num_subsets),
        }
    }
}

/// Configuration for SIRT-based reconstruction.
#[derive(Debug, Clone)]
pub struct SirtConfig {
    /// Algorithm to use (SIRT, ART, OSEM).
    pub algorithm: SirtAlgorithm,
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Relaxation parameter λ ∈ (0, 1] (typical: 0.5).
    pub relaxation_factor: f64,
    /// Regularization configuration.
    pub regularization: RegularizationConfig,
    /// Convergence tolerance (relative residual).
    pub tolerance: f64,
    /// Minimum relative change to continue iterations.
    pub min_relative_change: f64,
    /// Enable convergence monitoring/logging.
    pub verbose: bool,
}

impl Default for SirtConfig {
    fn default() -> Self {
        Self {
            algorithm: SirtAlgorithm::Sirt,
            max_iterations: 100,
            relaxation_factor: 0.5,
            regularization: RegularizationConfig::default(),
            tolerance: 1e-6,
            min_relative_change: 1e-8,
            verbose: false,
        }
    }
}

impl SirtConfig {
    /// Select SIRT algorithm.
    pub fn with_sirt(mut self) -> Self {
        self.algorithm = SirtAlgorithm::Sirt;
        self
    }

    /// Select ART algorithm.
    pub fn with_art(mut self) -> Self {
        self.algorithm = SirtAlgorithm::Art;
        self
    }

    /// Select OSEM algorithm with `num_subsets` subsets.
    pub fn with_osem(mut self, num_subsets: usize) -> Self {
        self.algorithm = SirtAlgorithm::Osem { num_subsets };
        self
    }

    /// Set maximum number of iterations.
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.max_iterations = iterations;
        self
    }

    /// Set relaxation factor (clamped to `[0.001, 1.0]`).
    pub fn with_relaxation(mut self, factor: f64) -> Self {
        self.relaxation_factor = factor.clamp(0.001, 1.0);
        self
    }

    /// Set regularization configuration.
    pub fn with_regularization(mut self, reg: RegularizationConfig) -> Self {
        self.regularization = reg;
        self
    }

    /// Enable or disable verbose logging.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

/// Result returned from a SIRT/ART/OSEM reconstruction.
#[derive(Debug, Clone)]
pub struct SirtResult {
    /// Reconstructed image.
    pub image: Array3<f64>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Final residual norm `‖Ax − b‖`.
    pub final_residual: f64,
    /// Residual norm at each iteration (for convergence analysis).
    pub residual_history: Vec<f64>,
    /// Whether convergence criteria were satisfied.
    pub converged: bool,
    /// Wall-clock computation time in seconds.
    pub computation_time: f64,
}
