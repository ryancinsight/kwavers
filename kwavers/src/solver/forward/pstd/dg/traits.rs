//! DG adaptation traits for discontinuity detection, solution coupling, modal
//! projection, reconstruction, and solver orchestration.
//!
//! These traits define the PSTD/DG boundary without binding acoustic kernels to
//! concrete coupling or limiter implementations.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::Array3;

/// Trait for discontinuity detection
pub trait DiscontinuityDetection: Send + Sync {
    /// Detect discontinuities in the field
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn detect(&self, field: &Array3<f64>, grid: &Grid) -> KwaversResult<Array3<bool>>;

    /// Update detection threshold
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn update_threshold(&mut self, threshold: f64);
}

/// Trait for solution coupling between different methods
pub trait SolutionCoupling: Send + Sync {
    /// Couple two solutions, writing the blended result into `output`.
    ///
    /// `output` does NOT need to be pre-initialized; every element is written.
    /// This is the zero-allocation hot-path variant.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn couple_into(
        &self,
        solution1: &Array3<f64>,
        solution2: &Array3<f64>,
        mask: &Array3<bool>,
        original: &Array3<f64>,
        output: &mut Array3<f64>,
    ) -> KwaversResult<()>;

    /// Convenience wrapper — allocates and returns the coupled field.
    /// Prefer [`Self::couple_into`] in time-step loops.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    fn couple(
        &self,
        solution1: &Array3<f64>,
        solution2: &Array3<f64>,
        mask: &Array3<bool>,
        original: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let mut out = original.clone();
        self.couple_into(solution1, solution2, mask, original, &mut out)?;
        Ok(out)
    }
}

/// Trait for DG-specific operations
pub trait DGOperations: Send + Sync {
    /// Compute numerical flux at element interfaces
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn compute_flux(&self, left_state: f64, right_state: f64, normal: f64) -> f64;

    /// Project field onto DG basis functions
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn project_to_basis(&self, field: &Array3<f64>) -> KwaversResult<Array3<f64>>;

    /// Reconstruct field from DG basis coefficients
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn reconstruct_from_basis(&self, coefficients: &Array3<f64>) -> KwaversResult<Array3<f64>>;
}

/// Trait for numerical solvers
pub trait NumericalSolver: Send + Sync {
    /// Advance the field one time step `dt` subject to the active-cell mask.
    ///
    /// # Errors
    /// - Returns [`Err`] if the numerical scheme diverges or a dimension mismatch occurs.
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
