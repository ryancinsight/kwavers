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
// pub mod shock_detector; // Deleted
pub mod spectral_solver;
// pub mod tests; // Deleted
pub mod traits;

// Re-exports for convenience
pub use basis::BasisType;
pub use config::DGConfig;
pub use coupling::HybridCoupler;
pub use dg_solver::core::DGSolver;
pub use discontinuity_detector::DiscontinuityDetector;
pub use flux::{FluxType, LimiterType};
pub use traits::{DGOperations, DiscontinuityDetection, NumericalSolver, SolutionCoupling};

use crate::grid::Grid;
use crate::solver::constants::{
    CONSERVATION_TOLERANCE, DEFAULT_POLYNOMIAL_ORDER, DISCONTINUITY_THRESHOLD,
};

use ndarray::Array3;
use spectral_solver::RegionSpectralSolver;
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
    spectral_solver: RegionSpectralSolver,
    dg_solver: DGSolver,
    coupler: HybridCoupler,
    /// Discontinuity mask for hybrid approach
    discontinuity_mask: Option<Array3<bool>>,
    // Shock detector for shock capturing
    // shock_detector: Option<shock_capturing::ShockDetector>,
    // WENO limiter for shock regions
    // weno_limiter: Option<shock_capturing::WENOLimiter>,
    // Artificial viscosity for stabilization
    // artificial_viscosity: Option<shock_capturing::ArtificialViscosity>,
}

impl HybridSpectralDGSolver {
    /// Create a new Hybrid Spectral-DG solver
    pub fn new(config: HybridSpectralDGConfig, grid: Arc<Grid>) -> Self {
        let detector = DiscontinuityDetector::new(config.discontinuity_threshold);
        let spectral_solver = RegionSpectralSolver::new(config.spectral_order, grid.clone());
        let dg_config = DGConfig {
            polynomial_order: config.dg_polynomial_order,
            basis_type: BasisType::Legendre,
            flux_type: FluxType::LaxFriedrichs,
            use_limiter: true,
            limiter_type: LimiterType::Minmod,
            shock_threshold: config.discontinuity_threshold,
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
            // shock_detector: None,
            // weno_limiter: None,
            // artificial_viscosity: None,
        }
    }
}
