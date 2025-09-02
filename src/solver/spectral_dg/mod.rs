//! Spectral and Discontinuous Galerkin (DG) solver module
//!
//! This module provides high-order spectral methods and discontinuous Galerkin
//! methods for solving acoustic wave equations.

pub mod basis;
pub mod config;
pub mod coupling;
pub mod dg_solver;
pub mod discontinuity_detector;
pub mod flux;
pub mod matrices;
pub mod quadrature;
pub mod shock_capturing;
pub mod shock_detector;
pub mod spectral_solver;
pub mod tests;
pub mod traits;

// Re-exports for convenience
pub use basis::BasisType;
pub use config::DGConfig;
pub use coupling::HybridCoupler;
pub use dg_solver::DGSolver;
pub use discontinuity_detector::DiscontinuityDetector;
pub use flux::{FluxType, LimiterType};
pub use spectral_solver::SpectralSolver;
pub use traits::{DGOperations, DiscontinuityDetection, NumericalSolver, SolutionCoupling};

use crate::error::{KwaversError, PhysicsError};
use crate::grid::Grid;
use crate::solver::constants::*;
use crate::KwaversResult;
use ndarray::Array3;
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
    detector: DiscontinuityDetector,
    spectral_solver: SpectralSolver,
    dg_solver: DGSolver,
    coupler: HybridCoupler,
    /// Discontinuity mask for hybrid approach
    discontinuity_mask: Option<Array3<bool>>,
    /// Shock detector for shock capturing
    shock_detector: Option<shock_capturing::ShockDetector>,
    /// WENO limiter for shock regions
    weno_limiter: Option<shock_capturing::WENOLimiter>,
    /// Artificial viscosity for stabilization
    artificial_viscosity: Option<shock_capturing::ArtificialViscosity>,
}

impl HybridSpectralDGSolver {
    /// Create a new Hybrid Spectral-DG solver
    pub fn new(config: HybridSpectralDGConfig, grid: Arc<Grid>) -> Self {
        let detector = DiscontinuityDetector::new(config.discontinuity_threshold);
        let spectral_solver = SpectralSolver::new(config.spectral_order, grid.clone());
        let dg_config = DGConfig {
            polynomial_order: config.dg_polynomial_order,
            basis_type: BasisType::Legendre,
            flux_type: FluxType::LaxFriedrichs,
            use_limiter: true,
            limiter_type: LimiterType::Minmod,
        };
        let dg_solver = DGSolver::new(dg_config, grid.clone()).expect("Failed to create DG solver");
        let coupler = HybridCoupler::new(config.conservation_tolerance);

        Self {
            config,
            detector,
            spectral_solver,
            dg_solver,
            coupler,
            discontinuity_mask: None,
            shock_detector: None,
            weno_limiter: None,
            artificial_viscosity: None,
        }
    }

    /// Enable shock detection
    pub fn with_shock_detection(&mut self) -> &mut Self {
        self.shock_detector = Some(shock_capturing::ShockDetector::default());
        self
    }

    /// Enable WENO limiting with specified order (3, 5, or 7)
    pub fn with_weno_limiting(&mut self, order: usize) -> KwaversResult<&mut Self> {
        self.weno_limiter = Some(shock_capturing::WENOLimiter::new(order)?);
        Ok(self)
    }

    /// Enable artificial viscosity
    pub fn with_artificial_viscosity(&mut self) -> &mut Self {
        self.artificial_viscosity = Some(shock_capturing::ArtificialViscosity::default());
        self
    }

    /// Update the solution using the hybrid method
    pub fn solve(
        &mut self,
        field: &Array3<f64>,
        dt: f64,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        // Step 1: Detect discontinuities
        let discontinuity_mask = self.detector.detect(field, grid)?;
        self.discontinuity_mask = Some(discontinuity_mask.clone());

        // Step 2: Apply appropriate solver in each region
        let result = if self.config.adaptive_switching {
            // Use spectral solver in smooth regions
            let spectral_result = self.spectral_solver.solve(field, dt, &discontinuity_mask)?;

            // Use DG solver in discontinuous regions
            let dg_result = self.dg_solver.solve(field, dt, &discontinuity_mask)?;

            // Step 3: Couple the solutions ensuring conservation
            self.coupler
                .couple(&spectral_result, &dg_result, &discontinuity_mask, field)?
        } else {
            // Use only spectral solver if adaptive switching is disabled
            self.spectral_solver
                .solve(field, dt, &Array3::from_elem(field.dim(), false))?
        };

        // Step 4: Verify conservation properties
        self.verify_conservation(field, &result)?;

        Ok(result)
    }

    /// Verify that conservation properties are maintained
    fn verify_conservation(
        &self,
        initial: &Array3<f64>,
        final_field: &Array3<f64>,
    ) -> KwaversResult<()> {
        let initial_integral: f64 = initial.sum();
        let final_integral: f64 = final_field.sum();
        let conservation_error = (final_integral - initial_integral).abs()
            / initial_integral.abs().max(ABSOLUTE_TOLERANCE);

        if conservation_error > self.config.conservation_tolerance {
            log::warn!(
                "Conservation error {} exceeds tolerance {}",
                conservation_error,
                self.config.conservation_tolerance
            );
            return Err(KwaversError::Physics(PhysicsError::ConservationViolation {
                quantity: "mass".to_string(),
                initial: initial_integral,
                current: final_integral,
                tolerance: self.config.conservation_tolerance,
            }));
        }

        Ok(())
    }

    /// Get the current discontinuity mask
    pub fn discontinuity_mask(&self) -> Option<&Array3<bool>> {
        self.discontinuity_mask.as_ref()
    }

    /// Update configuration
    pub fn update_config(&mut self, config: HybridSpectralDGConfig) {
        self.config = config;
        self.detector
            .update_threshold(self.config.discontinuity_threshold);
        self.spectral_solver
            .update_order(self.config.spectral_order);
        self.dg_solver.update_order(self.config.dg_polynomial_order);
    }
}
