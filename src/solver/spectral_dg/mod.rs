//! Hybrid Spectral-DG Methods Module
//! 
//! This module implements hybrid spectral and discontinuous Galerkin methods
//! for robust shock handling and high-order accuracy in smooth regions.
//! 
//! ## Literature References
//! 
//! 1. **Hesthaven, J. S., & Warburton, T. (2008)**. "Nodal discontinuous Galerkin 
//!    methods: algorithms, analysis, and applications." *Springer Science & Business 
//!    Media*. DOI: 10.1007/978-0-387-72067-8
//!    - Comprehensive DG theory and implementation
//!    - High-order polynomial bases
//! 
//! 2. **Persson, P. O., & Peraire, J. (2006)**. "Sub-cell shock capturing for 
//!    discontinuous Galerkin methods." *44th AIAA Aerospace Sciences Meeting and 
//!    Exhibit* (p. 112). DOI: 10.2514/6.2006-112
//!    - Shock detection algorithms
//!    - Artificial viscosity methods
//! 
//! 3. **Krivodonova, L., Xin, J., Remacle, J. F., Chevaugeon, N., & Flaherty, J. E. 
//!    (2004)**. "Shock detection and limiting with discontinuous Galerkin methods 
//!    for hyperbolic conservation laws." *Applied Numerical Mathematics*, 48(3-4), 
//!    323-338. DOI: 10.1016/j.apnum.2003.11.002
//!    - Discontinuity indicators
//!    - Limiting strategies
//! 
//! 4. **Cockburn, B., & Shu, C. W. (2001)**. "Runge–Kutta discontinuous Galerkin 
//!    methods for convection-dominated problems." *Journal of Scientific Computing*, 
//!    16(3), 173-261. DOI: 10.1023/A:1012873910884
//!    - Time integration for DG methods
//!    - Stability analysis
//! 
//! 5. **Gassner, G., Staudenmaier, M., Hindenlang, F., Atak, M., & Munz, C. D. 
//!    (2015)**. "A space–time adaptive discontinuous Galerkin scheme." *Computers & 
//!    Fluids*, 117, 247-261. DOI: 10.1016/j.compfluid.2015.05.002
//!    - Hybrid spectral-DG approaches
//!    - Adaptive method switching
//! 
//! ## Design Principles
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
pub mod enhanced_shock_handling;

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
    /// Enhanced shock detector for advanced shock capturing
    enhanced_detector: Option<enhanced_shock_handling::EnhancedShockDetector>,
    /// WENO limiter for shock regions
    weno_limiter: Option<enhanced_shock_handling::WENOLimiter>,
    /// Artificial viscosity for stabilization
    artificial_viscosity: Option<enhanced_shock_handling::ArtificialViscosity>,
}

impl HybridSpectralDGSolver {
    /// Create a new Hybrid Spectral-DG solver
    pub fn new(config: HybridSpectralDGConfig, grid: Arc<Grid>) -> Self {
        let detector = DiscontinuityDetector::new(config.discontinuity_threshold);
        let spectral_solver = SpectralSolver::new(config.spectral_order, grid.clone());
        let dg_config = dg_solver::DGConfig {
            polynomial_order: config.dg_polynomial_order,
            basis_type: dg_solver::BasisType::Legendre,
            flux_type: dg_solver::FluxType::LaxFriedrichs,
            use_limiter: true,
            limiter_type: dg_solver::LimiterType::MinMod,
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
            enhanced_detector: None,
            weno_limiter: None,
            artificial_viscosity: None,
        }
    }
    
    /// Enable enhanced shock detection
    pub fn with_shock_detection(&mut self) -> &mut Self {
        self.enhanced_detector = Some(enhanced_shock_handling::EnhancedShockDetector::default());
        self
    }
    
    /// Enable WENO limiting with specified order (3, 5, or 7)
    pub fn with_weno_limiting(&mut self, order: usize) -> KwaversResult<&mut Self> {
        self.weno_limiter = Some(enhanced_shock_handling::WENOLimiter::new(order)?);
        Ok(self)
    }
    
    /// Enable artificial viscosity
    pub fn with_artificial_viscosity(&mut self) -> &mut Self {
        self.artificial_viscosity = Some(enhanced_shock_handling::ArtificialViscosity::default());
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
            self.coupler.couple(
                &spectral_result,
                &dg_result,
                &discontinuity_mask,
                field,
            )?
        } else {
            // Use only spectral solver if adaptive switching is disabled
            self.spectral_solver.solve(field, dt, &Array3::from_elem(field.dim(), false))?
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

