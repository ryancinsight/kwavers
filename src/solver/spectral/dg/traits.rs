// src/physics/traits.rs stub/reconstruction

use crate::error::KwaversResult;
use crate::grid::Grid;
use ndarray::Array3;

/// Trait for discontinuity detection
pub trait DiscontinuityDetection: Send + Sync {
    /// Detect discontinuities in the field
    fn detect(&self, field: &Array3<f64>, grid: &Grid) -> KwaversResult<Array3<bool>>;

    /// Update detection threshold
    fn update_threshold(&mut self, threshold: f64);
}

/// Trait for solution coupling between different methods
pub trait SolutionCoupling: Send + Sync {
    /// Couple two solutions from different methods
    fn couple(
        &self,
        solution1: &Array3<f64>,
        solution2: &Array3<f64>,
        mask: &Array3<bool>,
        original: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>>;
}

/// Trait for DG-specific operations
pub trait DGOperations: Send + Sync {
    /// Compute numerical flux at element interfaces
    fn compute_flux(&self, left_state: f64, right_state: f64, normal: f64) -> f64;

    /// Project field onto DG basis functions
    fn project_to_basis(&self, field: &Array3<f64>) -> KwaversResult<Array3<f64>>;

    /// Reconstruct field from DG basis coefficients
    fn reconstruct_from_basis(&self, coefficients: &Array3<f64>) -> KwaversResult<Array3<f64>>;
}

/// Trait for numerical solvers
pub trait NumericalSolver: Send + Sync {
    fn solve(
        &mut self,
        field: &Array3<f64>,
        dt: f64,
        mask: &Array3<bool>,
    ) -> KwaversResult<Array3<f64>>;

    fn max_stable_dt(&self, grid: &Grid) -> f64;

    fn update_order(&mut self, order: usize);
}

/// Trait for spectral operations
pub trait SpectralOperations: Send + Sync {
    // Add necessary methods
}
