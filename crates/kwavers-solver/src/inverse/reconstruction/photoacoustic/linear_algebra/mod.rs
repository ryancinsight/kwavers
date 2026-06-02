//! Robust linear algebra solvers for photoacoustic reconstruction.
//!
//! Provides numerically stable solvers for the inverse problems in photoacoustic
//! imaging, partitioned by regularization family:
//!
//! - `tikhonov` — Tikhonov regularized least squares (CGNE).
//! - `tv_l1` — Total Variation (ISTA) and L1/Lasso (FISTA) regularization.
//! - `svd` — Truncated SVD via power iteration.

mod svd;
mod tikhonov;
mod tv_l1;

/// Linear algebra solver with various regularization methods.
#[derive(Debug)]
pub struct PhotoacousticLinearSolver {
    pub(super) max_iterations: usize,
    pub(super) tolerance: f64,
}

impl PhotoacousticLinearSolver {
    /// Create a new linear solver with default parameters
    /// (`max_iterations = 1000`, `tolerance = 1e-8`).
    pub fn new() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-8,
        }
    }
}
