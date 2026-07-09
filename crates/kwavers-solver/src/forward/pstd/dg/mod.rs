//! Spectral and Discontinuous Galerkin (DG) solver module
//!
//! This module provides high-order spectral methods and discontinuous Galerkin
//! methods for solving acoustic wave equations.

pub mod basis;
pub mod config;
pub mod coupling;
pub mod cpml;
pub mod dg_solver;
pub mod discontinuity_detector;
pub mod flux;
pub mod matrices;
pub mod quadrature;
pub mod shock_capturing;
// pub mod shock_detector; // Deleted
pub mod spectral_solver;
// pub mod tests; // Deleted
pub mod traits;

// Re-exports for convenience
pub use basis::BasisType;
pub use config::{DGConfig, DgBoundaryCondition, DgTimeIntegrator, ShockCaptureConfig, WenoDegree};
pub use coupling::HybridCoupler;
pub use cpml::{DgCpmlAxis, DgCpmlConfig, DgCpmlMemoryWorkspace, DgCpmlProfiles};
pub use dg_solver::core::DGSolver;
pub use discontinuity_detector::DiscontinuityDetector;
pub use flux::{FluxType, LimiterType};
pub use traits::{DGOperations, DiscontinuityDetection, NumericalSolver, SolutionCoupling};

use crate::constants::{CONSERVATION_TOLERANCE, DEFAULT_POLYNOMIAL_ORDER, DISCONTINUITY_THRESHOLD};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;

use leto::Array3;
use spectral_solver::RegionPSTDSolver;
use std::sync::Arc;

/// Configuration for the Hybrid Spectral-DG solver
#[derive(Debug, Clone)]
pub struct HybridSpectralDGConfig {
    /// Threshold for discontinuity detection
    pub discontinuity_threshold: f64,
    /// Order of spectral method
    pub spectral_order: usize,
    /// Order of DG polynomial basis
    pub dg_polynomial_order: usize,
    /// Enable adaptive switching
    pub adaptive_switching: bool,
    /// Conservation tolerance
    pub conservation_tolerance: f64,
}

impl Default for HybridSpectralDGConfig {
    fn default() -> Self {
        Self {
            discontinuity_threshold: DISCONTINUITY_THRESHOLD,
            spectral_order: DEFAULT_POLYNOMIAL_ORDER,
            dg_polynomial_order: 3,
            adaptive_switching: true,
            conservation_tolerance: CONSERVATION_TOLERANCE,
        }
    }
}

/// Main Hybrid Spectral-DG solver
#[derive(Debug)]
pub struct HybridSpectralDGSolver {
    config: HybridSpectralDGConfig,
    grid: Arc<Grid>,
    detector: DiscontinuityDetector,
    spectral_solver: RegionPSTDSolver,
    dg_solver: DGSolver,
    coupler: HybridCoupler,
    /// Discontinuity mask for hybrid approach
    discontinuity_mask: Array3<bool>,
    spectral_mask: Array3<bool>,
    spectral_field: Array3<f64>,
    dg_field: Array3<f64>,
    has_discontinuity_mask: bool,
    // Shock detector for shock capturing
    // shock_detector: Option<shock_capturing::ShockDetector>,
    // WENO limiter for shock regions
    // weno_limiter: Option<shock_capturing::WENOLimiter>,
    // Artificial viscosity for stabilization
    // artificial_viscosity: Option<shock_capturing::ArtificialViscosity>,
}

impl HybridSpectralDGSolver {
    /// Create a new Hybrid Spectral-DG solver
    /// # Panics
    /// - Panics if `Failed to create DG solver`.
    ///
    pub fn new(config: HybridSpectralDGConfig, grid: Arc<Grid>) -> Self {
        let detector = DiscontinuityDetector::new(config.discontinuity_threshold);
        let spectral_solver = RegionPSTDSolver::new(config.spectral_order, grid.clone());
        let dg_config = DGConfig {
            polynomial_order: config.dg_polynomial_order,
            basis_type: BasisType::Legendre,
            flux_type: FluxType::LaxFriedrichs,
            use_limiter: true,
            limiter_type: LimiterType::Minmod,
            shock_threshold: config.discontinuity_threshold,
            ..DGConfig::default()
        };
        let dg_solver = DGSolver::new(dg_config, grid.clone()).expect("Failed to create DG solver");
        let coupler = HybridCoupler::new(config.conservation_tolerance);
        let shape = (grid.nx, grid.ny, grid.nz);

        Self {
            config,
            grid,
            detector,
            spectral_solver,
            dg_solver,
            coupler,
            discontinuity_mask: Array3::from_elem(shape, false),
            spectral_mask: Array3::from_elem(shape, true),
            spectral_field: Array3::zeros(shape),
            dg_field: Array3::zeros(shape),
            has_discontinuity_mask: false,
            // shock_detector: None,
            // weno_limiter: None,
            // artificial_viscosity: None,
        }
    }

    pub fn config(&self) -> &HybridSpectralDGConfig {
        &self.config
    }

    pub fn detector(&self) -> &DiscontinuityDetector {
        &self.detector
    }

    pub fn spectral_solver(&self) -> &RegionPSTDSolver {
        &self.spectral_solver
    }

    pub fn dg_solver(&self) -> &DGSolver {
        &self.dg_solver
    }

    pub fn coupler(&self) -> &HybridCoupler {
        &self.coupler
    }

    pub fn discontinuity_mask(&self) -> Option<&Array3<bool>> {
        if self.has_discontinuity_mask {
            Some(&self.discontinuity_mask)
        } else {
            None
        }
    }

    /// Advance one hybrid Spectral-DG step into caller-owned output.
    ///
    /// Smooth cells are advanced by the spectral wave step. Cells flagged by the
    /// discontinuity detector are advanced by the DG step and coupled back through
    /// `HybridCoupler`.
    ///
    /// # Errors
    /// Returns an error when dimensions, wave speed, or solver internals violate
    /// their contracts.
    pub fn solve_step_into(
        &mut self,
        field: &Array3<f64>,
        dt: f64,
        c: f64,
        output: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let shape = (self.grid.nx, self.grid.ny, self.grid.nz);
        if field.dim() != shape || output.dim() != shape {
            return Err(KwaversError::InvalidInput(format!(
                "HybridSpectralDGSolver dimension mismatch: field={:?}, output={:?}, grid={shape:?}",
                field.dim(),
                output.dim()
            )));
        }

        self.detector
            .detect_into(field, &self.grid, &mut self.discontinuity_mask)?;
        self.has_discontinuity_mask = true;
        for (smooth, &discontinuous) in self
            .spectral_mask
            .iter_mut()
            .zip(self.discontinuity_mask.iter())
        {
            *smooth = !discontinuous;
        }

        self.spectral_solver.spectral_wave_step_into(
            field,
            dt,
            c,
            &self.spectral_mask,
            &mut self.spectral_field,
        )?;
        self.dg_solver
            .solve_into(field, dt, &self.discontinuity_mask, &mut self.dg_field)?;
        self.coupler.couple_into(
            &self.spectral_field,
            &self.dg_field,
            &self.discontinuity_mask,
            field,
            output,
        )?;
        Ok(())
    }

    /// Allocating wrapper for one hybrid Spectral-DG step.
    ///
    /// Prefer [`Self::solve_step_into`] inside time loops.
    pub fn solve_step(
        &mut self,
        field: &Array3<f64>,
        dt: f64,
        c: f64,
    ) -> KwaversResult<Array3<f64>> {
        let mut output = Array3::zeros(field.dim());
        self.solve_step_into(field, dt, c, &mut output)?;
        Ok(output)
    }
}

impl NumericalSolver for HybridSpectralDGSolver {
    fn solve(
        &mut self,
        field: &Array3<f64>,
        dt: f64,
        mask: &Array3<bool>,
    ) -> KwaversResult<Array3<f64>> {
        if field.dim() != mask.dim() {
            return Err(KwaversError::InvalidInput(format!(
                "HybridSpectralDGSolver mask shape {:?} does not match field {:?}",
                mask.dim(),
                field.dim()
            )));
        }
        let mut output = Array3::zeros(field.dim());
        self.solve_step_into(field, dt, self.dg_solver.config().sound_speed, &mut output)?;
        for ((out, &active), &input) in output.iter_mut().zip(mask.iter()).zip(field.iter()) {
            if !active {
                *out = input;
            }
        }
        Ok(output)
    }

    fn max_stable_dt(&self, grid: &Grid) -> f64 {
        self.dg_solver.max_stable_dt(grid)
    }

    fn update_order(&mut self, order: usize) {
        self.config.spectral_order = order;
        self.config.dg_polynomial_order = order;
        self.dg_solver.update_order(order);
    }
}

#[cfg(test)]
mod tests {
    use super::{HybridSpectralDGConfig, HybridSpectralDGSolver};
    use kwavers_grid::Grid;
    use leto::Array3;
    use std::sync::Arc;

    #[test]
    fn hybrid_solver_runs_1d_2d_3d_with_reused_workspaces() {
        for dims in [(4, 1, 1), (4, 4, 1), (4, 4, 4)] {
            let grid = Arc::new(Grid::new(dims.0, dims.1, dims.2, 1.0, 1.0, 1.0).unwrap());
            let config = HybridSpectralDGConfig {
                discontinuity_threshold: 0.25,
                spectral_order: 2,
                dg_polynomial_order: 1,
                adaptive_switching: true,
                conservation_tolerance: 1.0e-12,
            };
            let mut solver = HybridSpectralDGSolver::new(config, Arc::clone(&grid));
            let field = Array3::from_shape_fn(dims, |(i, j, k)| {
                (i as f64 + 0.5 * j as f64 + 0.25 * k as f64).sin()
            });
            let mut output = Array3::zeros(dims);

            let spectral_ptr = solver.spectral_field.as_ptr();
            let dg_ptr = solver.dg_field.as_ptr();
            let mask_ptr = solver.discontinuity_mask.as_ptr();

            solver
                .solve_step_into(&field, 1.0e-5, 1.0, &mut output)
                .unwrap();
            let second = output.clone();
            solver
                .solve_step_into(&second, 1.0e-5, 1.0, &mut output)
                .unwrap();

            assert_eq!(solver.spectral_field.as_ptr(), spectral_ptr);
            assert_eq!(solver.dg_field.as_ptr(), dg_ptr);
            assert_eq!(solver.discontinuity_mask.as_ptr(), mask_ptr);
            assert!(output.iter().all(|value| value.is_finite()));
            assert!(solver.discontinuity_mask().is_some());
        }
    }
}
