//! Core hybrid PSTD/FDTD solver implementation

use crate::boundary::Boundary;
use crate::error::{KwaversError, KwaversResult};
use crate::grid::Grid;
use crate::medium::Medium;
use crate::solver::fdtd::FdtdSolver;
use crate::solver::hybrid::adaptive_selection::AdaptiveSelector;
use crate::solver::hybrid::config::{DecompositionStrategy, HybridConfig};
use crate::solver::hybrid::coupling::CouplingInterface;
use crate::solver::hybrid::domain_decomposition::{DomainDecomposer, DomainRegion, DomainType};
use crate::solver::hybrid::metrics::{HybridMetrics, ValidationResults};
use crate::solver::pstd::PstdSolver;
use crate::source::Source;
use log::{debug, info};
use ndarray::{s, Array4};
use std::time::Instant;

/// Context for regional solver application
struct RegionalContext<'a> {
    source: &'a dyn Source,
    boundary: &'a mut dyn Boundary,
}

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
    /// Update fields for hybrid solver
    pub fn update_fields(&mut self, _fields: &mut Array4<f64>, dt: f64) -> KwaversResult<()> {
        // Validate inputs
        if dt <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Time step must be positive".to_string(),
            ));
        }

        // Field update implementation
        // This coordinates PSTD and FDTD updates based on domain decomposition
        // Currently simplified for compilation

        Ok(())
    }
    /// Create a new hybrid solver
    pub fn new(config: HybridConfig, grid: &Grid) -> KwaversResult<Self> {
        info!("Initializing hybrid PSTD/FDTD solver");

        // Initialize component solvers
        let pstd_solver = PstdSolver::new(config.pstd_config.clone(), grid)?;
        let fdtd_solver = FdtdSolver::new(config.fdtd_config, grid)?;

        // Initialize domain decomposition
        let decomposer = DomainDecomposer::new();
        let selector = AdaptiveSelector::new(config.selection_criteria.clone());
        let coupling = CouplingInterface::new(
            grid,
            grid,
            crate::solver::hybrid::coupling::InterpolationScheme::Linear,
        )?;

        // Perform initial domain decomposition
        // Create a default medium for initial decomposition
        let default_medium = crate::medium::homogeneous::HomogeneousMedium::water(grid);
        let regions = decomposer.decompose(grid, &default_medium)?;

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
        _source: &dyn Source,
        _boundary: &mut dyn Boundary,
        dt: f64,
        t: f64,
    ) -> KwaversResult<()> {
        let start = Instant::now();

        // Update domain decomposition if dynamic
        if self.config.decomposition_strategy == DecompositionStrategy::Dynamic {
            self.update_decomposition(fields, medium)?;
        }

        // Process each region with appropriate solver
        let regions = self.regions.clone();
        for region in &regions {
            match region.domain_type {
                DomainType::PSTD => {
                    let pstd_start = Instant::now();
                    self.apply_pstd_region(fields, medium, dt, t, region)?;
                    self.metrics.pstd_time += pstd_start.elapsed();
                }
                DomainType::FDTD => {
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
        debug!(
            "Hybrid solver step {} completed in {:?}",
            self.time_step,
            start.elapsed()
        );

        Ok(())
    }

    /// Apply PSTD solver to a region
    fn apply_pstd_region(
        &mut self,
        fields: &mut Array4<f64>,
        _medium: &dyn Medium,
        _dt: f64,
        _t: f64,
        region: &DomainRegion,
    ) -> KwaversResult<()> {
        // Extract region view
        let mut region_fields = fields.slice_mut(s![
            ..,
            region.start.0..region.end.0,
            region.start.1..region.end.1,
            region.start.2..region.end.2,
        ]);

        // Apply PSTD update to the region
        // Convert region view to owned array for PSTD solver
        let region_array = region_fields.to_owned();

        // Update using PSTD solver with proper context
        // For now, we'll pass through the full solver call
        // A proper implementation would create regional views of source and boundary

        // Copy results back
        region_fields.assign(&region_array);

        debug!("Applied PSTD to region {:?}", region);
        Ok(())
    }

    /// Apply FDTD solver to a region
    fn apply_fdtd_region(
        &mut self,
        fields: &mut Array4<f64>,
        _medium: &dyn Medium,
        _dt: f64,
        _t: f64,
        region: &DomainRegion,
    ) -> KwaversResult<()> {
        // Extract region view
        let mut region_fields = fields.slice_mut(s![
            ..,
            region.start.0..region.end.0,
            region.start.1..region.end.1,
            region.start.2..region.end.2,
        ]);

        // Apply FDTD update to the region
        // Convert region view to owned array for FDTD solver
        let region_array = region_fields.to_owned();

        // Update using FDTD solver
        // Note: FDTD solver requires source and boundary which are not available in region context
        // This is a fundamental architectural issue that needs redesign

        // Copy results back
        region_fields.assign(&region_array);

        debug!("Applied FDTD to region {:?}", region);
        Ok(())
    }

    /// Apply hybrid processing to transition region
    fn apply_hybrid_region(
        &mut self,
        fields: &mut Array4<f64>,
        _medium: &dyn Medium,
        _dt: f64,
        _t: f64,
        region: &DomainRegion,
    ) -> KwaversResult<()> {
        // Apply blended approach in transition regions
        // This uses weighted averaging between PSTD and FDTD solutions

        let pstd_fields = fields
            .slice(s![
                ..,
                region.start.0..region.end.0,
                region.start.1..region.end.1,
                region.start.2..region.end.2,
            ])
            .to_owned();

        let fdtd_fields = pstd_fields.clone();

        // Apply both solvers with proper coordination
        // Note: Both solvers require source and boundary which are not available in region context
        // This hybrid approach needs fundamental redesign to properly coordinate solvers

        // Blend results with distance-based weighting
        const BLEND_WIDTH: usize = 5; // Grid points for smooth transition
        for i in 0..pstd_fields.shape()[1] {
            for j in 0..pstd_fields.shape()[2] {
                for k in 0..pstd_fields.shape()[3] {
                    // Calculate distance from region boundary
                    let dist_from_boundary = ((i.min(pstd_fields.shape()[1] - i - 1))
                        .min(j.min(pstd_fields.shape()[2] - j - 1))
                        .min(k.min(pstd_fields.shape()[3] - k - 1)))
                        as f64;

                    // Smooth blending function
                    let weight = if dist_from_boundary < BLEND_WIDTH as f64 {
                        0.5 * (1.0
                            + (std::f64::consts::PI * dist_from_boundary / BLEND_WIDTH as f64)
                                .cos())
                    } else {
                        1.0
                    };

                    // Apply weighted average
                    for field_idx in 0..pstd_fields.shape()[0] {
                        fields[[
                            field_idx,
                            region.start.0 + i,
                            region.start.1 + j,
                            region.start.2 + k,
                        ]] = weight * pstd_fields[[field_idx, i, j, k]]
                            + (1.0 - weight) * fdtd_fields[[field_idx, i, j, k]];
                    }
                }
            }
        }

        debug!("Applied hybrid blending to region {:?}", region);
        Ok(())
    }

    /// Apply coupling between regions
    fn apply_coupling(&mut self, fields: &mut Array4<f64>) -> KwaversResult<()> {
        self.coupling
            .apply_coupling(fields, &self.regions, &self.grid)
    }

    /// Update domain decomposition based on current fields
    fn update_decomposition(
        &mut self,
        fields: &Array4<f64>,
        medium: &dyn Medium,
    ) -> KwaversResult<()> {
        let start = Instant::now();

        // Re-analyze field characteristics
        self.selector.update_metrics(fields);

        // Update decomposition if needed
        let new_regions = self.decomposer.decompose(&self.grid, medium)?;

        if new_regions.len() != self.regions.len() {
            info!(
                "Domain decomposition updated: {} regions",
                new_regions.len()
            );
            self.regions = new_regions;
        }

        self.metrics.decomposition_time += start.elapsed();
        Ok(())
    }

    /// Validate solution quality
    fn validate_solution(&mut self, fields: &Array4<f64>, _time: f64) -> KwaversResult<()> {
        use crate::physics::field_mapping::UnifiedFieldType;

        // Check for NaN or infinite values
        let pressure = fields.index_axis(ndarray::Axis(0), UnifiedFieldType::Pressure.index());
        let has_nan = pressure.iter().any(|&x| x.is_nan());
        let has_inf = pressure.iter().any(|&x| x.is_infinite());

        if has_nan || has_inf {
            self.validation_results.quality_score = 0.0;
            self.validation_results.nan_inf_count += 1;

            if self.config.validation.check_nan_inf {
                return Err(KwaversError::Validation(
                    crate::error::ValidationError::FieldValidation {
                        field: "pressure".to_string(),
                        value: format!("NaN: {has_nan}, Inf: {has_inf}"),
                        constraint: "Must be finite".to_string(),
                    },
                ));
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
