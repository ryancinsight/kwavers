//! Core hybrid PSTD/FDTD solver implementation

use crate::grid::Grid;
use crate::medium::Medium;
use crate::error::{KwaversResult, KwaversError};
use crate::solver::pstd::{PstdSolver, PstdConfig};
use crate::solver::fdtd::{FdtdSolver, FdtdConfig};
use crate::solver::hybrid::domain_decomposition::{DomainDecomposer, DomainRegion, DomainType};
use crate::solver::hybrid::adaptive_selection::{AdaptiveSelector, SelectionCriteria};
use crate::solver::hybrid::coupling::{CouplingInterface, InterpolationScheme};
use crate::solver::hybrid::config::{HybridConfig, DecompositionStrategy};
use crate::solver::hybrid::metrics::{HybridMetrics, ValidationResults};
use crate::physics::field_mapping::UnifiedFieldType;
use ndarray::{Array4, s, Zip};
use std::collections::HashMap;
use std::time::Instant;
use log::{debug, info};

/// Hybrid PSTD/FDTD solver combining spectral and finite-difference methods
#[derive(Debug)]
pub struct HybridSolver {
    /// Configuration
    config: HybridConfig,
    
    /// Computational grid
    grid: Grid,
    
    /// PSTD solver for smooth regions
    pstd_solver: PstdSolver,
    
    /// FDTD solver for discontinuous regions
    fdtd_solver: FdtdSolver,
    
    /// Domain decomposer
    decomposer: DomainDecomposer,
    
    /// Adaptive selector for method choice
    selector: AdaptiveSelector,
    
    /// Coupling interface manager
    coupling: CouplingInterface,
    
    /// Current domain regions
    regions: Vec<DomainRegion>,
    
    /// Performance metrics
    metrics: HybridMetrics,
    
    /// Validation results
    validation_results: ValidationResults,
    
    /// Time step counter
    time_step: usize,
}

impl HybridSolver {
    /// Create a new hybrid solver
    pub fn new(config: HybridConfig, grid: &Grid) -> KwaversResult<Self> {
        info!("Initializing hybrid PSTD/FDTD solver");
        
        // Initialize component solvers
        let pstd_solver = PstdSolver::new(config.pstd_config.clone(), grid)?;
        let fdtd_solver = FdtdSolver::new(config.fdtd_config.clone(), grid)?;
        
        // Initialize domain decomposition
        let decomposer = DomainDecomposer::new(config.decomposition_strategy);
        let selector = AdaptiveSelector::new(config.selection_criteria.clone());
        let coupling = CouplingInterface::new(
            config.coupling_interface.interpolation_scheme,
            config.coupling_interface.ghost_cells,
        );
        
        // Perform initial domain decomposition
        let regions = decomposer.decompose(grid, &selector)?;
        
        info!("Hybrid solver initialized with {} regions", regions.len());
        
        Ok(Self {
            config,
            grid: grid.clone(),
            pstd_solver,
            fdtd_solver,
            decomposer,
            selector,
            coupling,
            regions,
            metrics: HybridMetrics::new(),
            validation_results: ValidationResults::default(),
            time_step: 0,
        })
    }
    
    /// Update fields for one time step
    pub fn update(
        &mut self,
        fields: &mut Array4<f64>,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
    ) -> KwaversResult<()> {
        let start = Instant::now();
        
        // Update domain decomposition if dynamic
        if self.config.decomposition_strategy == DecompositionStrategy::Dynamic {
            self.update_decomposition(fields, medium)?;
        }
        
        // Process each region with appropriate solver
        for region in &self.regions {
            match region.domain_type {
                DomainType::Pstd => {
                    let pstd_start = Instant::now();
                    self.apply_pstd_region(fields, medium, dt, t, region)?;
                    self.metrics.pstd_time += pstd_start.elapsed();
                }
                DomainType::Fdtd => {
                    let fdtd_start = Instant::now();
                    self.apply_fdtd_region(fields, medium, dt, t, region)?;
                    self.metrics.fdtd_time += fdtd_start.elapsed();
                }
                DomainType::Hybrid => {
                    // Process as transition region
                    self.apply_hybrid_region(fields, medium, dt, t, region)?;
                }
            }
        }
        
        // Apply coupling between regions
        let coupling_start = Instant::now();
        self.apply_coupling(fields)?;
        self.metrics.coupling_time += coupling_start.elapsed();
        
        // Validate solution if enabled
        if self.config.validation.enable_validation {
            self.validate_solution(fields, t)?;
        }
        
        self.time_step += 1;
        debug!("Hybrid solver step {} completed in {:?}", self.time_step, start.elapsed());
        
        Ok(())
    }
    
    /// Apply PSTD solver to a region
    fn apply_pstd_region(
        &mut self,
        fields: &mut Array4<f64>,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        region: &DomainRegion,
    ) -> KwaversResult<()> {
        // Extract region view
        let mut region_fields = fields.slice_mut(s![
            ..,
            region.start.0..region.end.0,
            region.start.1..region.end.1,
            region.start.2..region.end.2,
        ]);
        
        // Apply PSTD update
        // Note: This would need proper integration with PSTD solver's update method
        debug!("Applying PSTD to region {:?}", region);
        
        Ok(())
    }
    
    /// Apply FDTD solver to a region
    fn apply_fdtd_region(
        &mut self,
        fields: &mut Array4<f64>,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        region: &DomainRegion,
    ) -> KwaversResult<()> {
        // Extract region view
        let mut region_fields = fields.slice_mut(s![
            ..,
            region.start.0..region.end.0,
            region.start.1..region.end.1,
            region.start.2..region.end.2,
        ]);
        
        // Apply FDTD update
        // Note: This would need proper integration with FDTD solver's update method
        debug!("Applying FDTD to region {:?}", region);
        
        Ok(())
    }
    
    /// Apply hybrid processing to transition region
    fn apply_hybrid_region(
        &mut self,
        fields: &mut Array4<f64>,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        region: &DomainRegion,
    ) -> KwaversResult<()> {
        // Apply blended approach in transition regions
        debug!("Applying hybrid processing to region {:?}", region);
        Ok(())
    }
    
    /// Apply coupling between regions
    fn apply_coupling(&mut self, fields: &mut Array4<f64>) -> KwaversResult<()> {
        self.coupling.apply_coupling(fields, &self.regions, &self.grid)
    }
    
    /// Update domain decomposition based on current fields
    fn update_decomposition(&mut self, fields: &Array4<f64>, medium: &dyn Medium) -> KwaversResult<()> {
        let start = Instant::now();
        
        // Re-analyze field characteristics
        self.selector.update_metrics(fields, &self.grid)?;
        
        // Update decomposition if needed
        let new_regions = self.decomposer.decompose(&self.grid, medium)?;
        
        if new_regions.len() != self.regions.len() {
            info!("Domain decomposition updated: {} regions", new_regions.len());
            self.regions = new_regions;
        }
        
        self.metrics.decomposition_time += start.elapsed();
        Ok(())
    }
    
    /// Validate solution quality
    fn validate_solution(
        &mut self,
        fields: &Array4<f64>,
        time: f64,
    ) -> KwaversResult<()> {
        use crate::physics::field_mapping::UnifiedFieldType;
        
        // Check for NaN or infinite values
        let pressure = fields.index_axis(ndarray::Axis(0), UnifiedFieldType::Pressure.index());
        let has_nan = pressure.iter().any(|&x| x.is_nan());
        let has_inf = pressure.iter().any(|&x| x.is_infinite());
        
        if has_nan || has_inf {
            self.validation_results.quality_score = 0.0;
            self.validation_results.nan_inf_count += 1;
            
            if self.config.validation.check_nan_inf {
                return Err(KwaversError::Validation(crate::error::ValidationError::FieldValidation {
                    field: "pressure".to_string(),
                    value: format!("NaN: {}, Inf: {}", has_nan, has_inf),
                    constraint: "Must be finite".to_string(),
                }));
            }
        }
        
        Ok(())
    }
    
    /// Get performance metrics
    pub fn metrics(&self) -> &HybridMetrics {
        &self.metrics
    }
    
    /// Get validation results
    pub fn validation_results(&self) -> &ValidationResults {
        &self.validation_results
    }
}