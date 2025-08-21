//! Common traits for numerical solvers
//!
//! This module defines the common interfaces that all numerical solvers
//! must implement, following the Interface Segregation Principle (ISP).

use crate::grid::Grid;
use crate::KwaversResult;
use ndarray::Array3;

/// Configuration trait for numerical solvers
pub trait SolverConfig: Clone + Send + Sync {
    /// Get the order of accuracy
    fn order(&self) -> usize;

    /// Validate the configuration
    fn validate(&self) -> KwaversResult<()>;
}

/// Base trait for numerical solvers
pub trait NumericalSolver: Send + Sync {
    /// Solve the equation for one time step
    ///
    /// # Arguments
    /// * `field` - Input field to evolve
    /// * `dt` - Time step
    /// * `mask` - Optional mask indicating regions where this solver should be active
    ///
    /// # Returns
    /// The updated field after one time step
    fn solve(
        &self,
        field: &Array3<f64>,
        dt: f64,
        mask: &Array3<bool>,
    ) -> KwaversResult<Array3<f64>>;

    /// Get the maximum stable time step for this solver
    fn max_stable_dt(&self, grid: &Grid) -> f64;

    /// Update the solver order/accuracy
    fn update_order(&mut self, order: usize);
}

/// Trait for discontinuity detection
pub trait DiscontinuityDetection: Send + Sync {
    /// Detect discontinuities in the field
    ///
    /// # Arguments
    /// * `field` - Field to analyze
    /// * `grid` - Computational grid
    ///
    /// # Returns
    /// Boolean mask where true indicates a discontinuity
    fn detect(&self, field: &Array3<f64>, grid: &Grid) -> KwaversResult<Array3<bool>>;

    /// Update detection threshold
    fn update_threshold(&mut self, threshold: f64);
}

/// Trait for solution coupling between different methods
pub trait SolutionCoupling: Send + Sync {
    /// Couple two solutions from different methods
    ///
    /// # Arguments
    /// * `solution1` - First solution (e.g., from spectral method)
    /// * `solution2` - Second solution (e.g., from DG method)
    /// * `mask` - Mask indicating which solution to use where
    /// * `original` - Original field for conservation checks
    ///
    /// # Returns
    /// The coupled solution
    fn couple(
        &self,
        solution1: &Array3<f64>,
        solution2: &Array3<f64>,
        mask: &Array3<bool>,
        original: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>>;
}

/// Trait for spectral operations
pub trait SpectralOperations: Send + Sync {
    /// Compute spatial derivatives using spectral methods
    fn spectral_derivative(
        &self,
        field: &Array3<f64>,
        direction: usize,
    ) -> KwaversResult<Array3<f64>>;

    /// Apply spectral filter for de-aliasing
    fn apply_filter(&self, field: &mut Array3<f64>);
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
