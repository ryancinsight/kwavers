//! Hybrid Spectral-DG Methods Module
//! 
//! This module implements hybrid spectral and discontinuous Galerkin methods
//! for robust shock handling and high-order accuracy in smooth regions.
//! 
//! # Design Principles
//! - SOLID: Each component (spectral solver, DG solver, discontinuity detector) is a separate module
//! - CUPID: Composable solvers with clear interfaces
//! - GRASP: Clear separation of responsibilities between detection, solving, and coupling
//! - DRY: Shared numerical utilities and interfaces
//! - KISS: Simple switching logic based on discontinuity detection
//! - YAGNI: Only implementing validated numerical methods
//! - Clean: Comprehensive documentation and testing

pub mod discontinuity_detector;
pub mod spectral_solver;
pub mod dg_solver;
pub mod coupling;
pub mod traits;

#[cfg(test)]
mod tests;

// Re-export main types
pub use discontinuity_detector::DiscontinuityDetector;
pub use spectral_solver::SpectralSolver;
pub use dg_solver::DGSolver;
pub use coupling::HybridCoupler;
pub use traits::{NumericalSolver, SolverConfig, SolutionCoupling};

use crate::grid::Grid;
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
            discontinuity_threshold: 0.1,
            spectral_order: 8,
            dg_polynomial_order: 3,
            adaptive_switching: true,
            conservation_tolerance: 1e-10,
        }
    }
}

/// Main Hybrid Spectral-DG solver
pub struct HybridSpectralDGSolver {
    config: HybridSpectralDGConfig,
    detector: DiscontinuityDetector,
    spectral_solver: SpectralSolver,
    dg_solver: DGSolver,
    coupler: HybridCoupler,
    /// Mask indicating regions where DG should be used (true) vs spectral (false)
    discontinuity_mask: Option<Array3<bool>>,
}

impl HybridSpectralDGSolver {
    /// Create a new Hybrid Spectral-DG solver
    pub fn new(config: HybridSpectralDGConfig, grid: Arc<Grid>) -> Self {
        let detector = DiscontinuityDetector::new(config.discontinuity_threshold);
        let spectral_solver = SpectralSolver::new(config.spectral_order, grid.clone());
        let dg_solver = DGSolver::new(config.dg_polynomial_order, grid.clone());
        let coupler = HybridCoupler::new(config.conservation_tolerance);
        
        Self {
            config,
            detector,
            spectral_solver,
            dg_solver,
            coupler,
            discontinuity_mask: None,
        }
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
        let mut result = Array3::zeros(field.dim());
        
        if self.config.adaptive_switching {
            // Use spectral solver in smooth regions
            let spectral_result = self.spectral_solver.solve(field, dt, &discontinuity_mask)?;
            
            // Use DG solver in discontinuous regions
            let dg_result = self.dg_solver.solve(field, dt, &discontinuity_mask)?;
            
            // Step 3: Couple the solutions ensuring conservation
            result = self.coupler.couple(
                &spectral_result,
                &dg_result,
                &discontinuity_mask,
                field,
            )?;
        } else {
            // Use only spectral solver if adaptive switching is disabled
            result = self.spectral_solver.solve(field, dt, &Array3::from_elem(field.dim(), false))?;
        }
        
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
        let conservation_error = (final_integral - initial_integral).abs() / initial_integral.abs().max(1e-10);
        
        if conservation_error > self.config.conservation_tolerance {
            log::warn!(
                "Conservation error {} exceeds tolerance {}",
                conservation_error,
                self.config.conservation_tolerance
            );
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
        self.detector.update_threshold(self.config.discontinuity_threshold);
        self.spectral_solver.update_order(self.config.spectral_order);
        self.dg_solver.update_order(self.config.dg_polynomial_order);
    }
}

