//! Core hybrid PSTD/FDTD solver implementation

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::boundary::Boundary;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::source::GridSource;
use crate::domain::source::Source;
use crate::solver::forward::pstd::PSTDSource;
use crate::solver::forward::{FdtdSolver, PSTDSolver};
use crate::solver::hybrid::adaptive_selection::AdaptiveSelector;
use crate::solver::hybrid::config::{DecompositionStrategy, HybridConfig};
use crate::solver::hybrid::coupling::CouplingInterface;
use crate::solver::hybrid::domain_decomposition::{DomainDecomposer, DomainRegion, DomainType};
use crate::solver::hybrid::metrics::{HybridMetrics, ValidationResults};
use log::{debug, info};
use ndarray::{s, Array4};
use std::sync::Arc;
use std::time::Instant;
/// Context for regional solver application
#[allow(dead_code)]
struct RegionalContext<'a> {
    source: &'a dyn Source,
    boundary: &'a mut dyn Boundary,
}

use crate::domain::field::wave::WaveFields;
use crate::domain::medium::MaterialFields;

/// Hybrid PSTD/FDTD solver combining spectral and finite-difference methods
#[derive(Debug)]
pub struct HybridSolver {
    /// Configuration
    config: HybridConfig,

    /// Computational grid
    grid: Grid,

    /// PSTD solver for smooth regions
    #[allow(dead_code)]
    pstd_solver: PSTDSolver,

    /// FDTD solver for discontinuous regions
    #[allow(dead_code)]
    fdtd_solver: FdtdSolver,

    /// Material properties cache
    materials: MaterialFields,

    // Unified Fields
    fields: WaveFields,

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
    pub fn new(config: HybridConfig, grid: &Grid, medium: &dyn Medium) -> KwaversResult<Self> {
        info!("Initializing hybrid Spectral/FDTD solver");

        // Initialize component solvers
        let pstd_solver = PSTDSolver::new(
            config.pstd_config.clone(),
            grid.clone(),
            medium,
            PSTDSource::default(),
        )?;
        let fdtd_solver = FdtdSolver::new(
            config.fdtd_config.clone(),
            grid,
            medium,
            GridSource::default(),
        )?;

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
        let default_medium = crate::domain::medium::homogeneous::HomogeneousMedium::water(grid);
        let regions =
            decomposer.decompose(grid, &default_medium, config.decomposition_strategy.clone())?;

        info!("Hybrid solver initialized with {} regions", regions.len());

        // Initialize material properties
        let mut materials = MaterialFields::new((grid.nx, grid.ny, grid.nz));

        // Compute material properties
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                    materials.rho0[[i, j, k]] =
                        crate::domain::medium::density_at(medium, x, y, z, grid);
                    materials.c0[[i, j, k]] =
                        crate::domain::medium::sound_speed_at(medium, x, y, z, grid);
                }
            }
        }

        let shape = (grid.nx, grid.ny, grid.nz);

        Ok(Self {
            config,
            grid: grid.clone(),
            pstd_solver,
            fdtd_solver,
            materials,
            decomposer,
            selector,
            coupling,
            regions,
            metrics: HybridMetrics::new(),
            validation_results: ValidationResults::default(),
            time_step: 0,
            fields: WaveFields::new(shape),
        })
    }

    /// Perform a single time step
    pub fn step_forward(&mut self) -> KwaversResult<()> {
        // 1. Sync global fields to component solvers
        // No loop over field types, just copy
        self.pstd_solver.fields.p.assign(&self.fields.p);
        self.pstd_solver.fields.ux.assign(&self.fields.ux);
        self.pstd_solver.fields.uy.assign(&self.fields.uy);
        self.pstd_solver.fields.uz.assign(&self.fields.uz);

        self.fdtd_solver.fields.p.assign(&self.fields.p);
        self.fdtd_solver.fields.ux.assign(&self.fields.ux);
        self.fdtd_solver.fields.uy.assign(&self.fields.uy);
        self.fdtd_solver.fields.uz.assign(&self.fields.uz);

        // 2. Step Solvers
        self.pstd_solver.step_forward()?;
        self.fdtd_solver.step_forward()?;

        // 3. Blend results based on regions
        let regions = self.regions.clone();
        for region in &regions {
            match region.domain_type {
                DomainType::PSTD => {
                    let slice = s![
                        region.start.0..region.end.0,
                        region.start.1..region.end.1,
                        region.start.2..region.end.2
                    ];
                    self.fields
                        .p
                        .slice_mut(slice)
                        .assign(&self.pstd_solver.fields.p.slice(slice));
                    self.fields
                        .ux
                        .slice_mut(slice)
                        .assign(&self.pstd_solver.fields.ux.slice(slice));
                    self.fields
                        .uy
                        .slice_mut(slice)
                        .assign(&self.pstd_solver.fields.uy.slice(slice));
                    self.fields
                        .uz
                        .slice_mut(slice)
                        .assign(&self.pstd_solver.fields.uz.slice(slice));
                }
                DomainType::FDTD => {
                    let slice = s![
                        region.start.0..region.end.0,
                        region.start.1..region.end.1,
                        region.start.2..region.end.2
                    ];
                    self.fields
                        .p
                        .slice_mut(slice)
                        .assign(&self.fdtd_solver.fields.p.slice(slice));
                    self.fields
                        .ux
                        .slice_mut(slice)
                        .assign(&self.fdtd_solver.fields.ux.slice(slice));
                    self.fields
                        .uy
                        .slice_mut(slice)
                        .assign(&self.fdtd_solver.fields.uy.slice(slice));
                    self.fields
                        .uz
                        .slice_mut(slice)
                        .assign(&self.fdtd_solver.fields.uz.slice(slice));
                }
                DomainType::Hybrid => {
                    self.apply_hybrid_region_blended_internal(region)?;
                }
            }
        }

        // 4. Apply Coupling (Placeholder logic as coupling previously used Array4)
        // For now, we assume blending covers basic coupling or we need to adapt CouplingInterface
        // self.coupling.apply_coupling(...) -> needs refactor to work with Array3s or wrapper

        self.time_step += 1;
        Ok(())
    }

    fn apply_hybrid_region_blended_internal(&mut self, region: &DomainRegion) -> KwaversResult<()> {
        let nx = region.end.0 - region.start.0;
        let ny = region.end.1 - region.start.1;
        let nz = region.end.2 - region.start.2;
        const BLEND_WIDTH: usize = 5;

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let dist_from_boundary = ((i.min(nx - i - 1))
                        .min(j.min(ny - j - 1))
                        .min(k.min(nz - k - 1)))
                        as f64;
                    let weight = if dist_from_boundary < BLEND_WIDTH as f64 {
                        0.5 * (1.0
                            + (std::f64::consts::PI * dist_from_boundary / BLEND_WIDTH as f64)
                                .cos())
                    } else {
                        1.0
                    };
                    let gi = region.start.0 + i;
                    let gj = region.start.1 + j;
                    let gk = region.start.2 + k;

                    self.fields.p[[gi, gj, gk]] = weight * self.pstd_solver.fields.p[[gi, gj, gk]]
                        + (1.0 - weight) * self.fdtd_solver.fields.p[[gi, gj, gk]];
                    self.fields.ux[[gi, gj, gk]] = weight
                        * self.pstd_solver.fields.ux[[gi, gj, gk]]
                        + (1.0 - weight) * self.fdtd_solver.fields.ux[[gi, gj, gk]];
                    self.fields.uy[[gi, gj, gk]] = weight
                        * self.pstd_solver.fields.uy[[gi, gj, gk]]
                        + (1.0 - weight) * self.fdtd_solver.fields.uy[[gi, gj, gk]];
                    self.fields.uz[[gi, gj, gk]] = weight
                        * self.pstd_solver.fields.uz[[gi, gj, gk]]
                        + (1.0 - weight) * self.fdtd_solver.fields.uz[[gi, gj, gk]];
                }
            }
        }
        Ok(())
    }

    /// Update fields for one time step
    pub fn update(
        &mut self,
        fields: &mut Array4<f64>,
        medium: &dyn Medium,
        _source: &dyn Source,
        _boundary: &mut dyn Boundary,
        _dt: f64,
        t: f64,
    ) -> KwaversResult<()> {
        let update_start = Instant::now();

        // Update domain decomposition if dynamic
        if self.config.decomposition_strategy == DecompositionStrategy::Dynamic {
            self.update_decomposition(fields, medium)?;
        }

        // 1. Sync global fields to component solvers
        use crate::domain::field::mapping::UnifiedFieldType;
        let p_idx = UnifiedFieldType::Pressure.index();
        let vx_idx = UnifiedFieldType::VelocityX.index();
        let vy_idx = UnifiedFieldType::VelocityY.index();
        let vz_idx = UnifiedFieldType::VelocityZ.index();

        // Sync PSTD Solver
        self.pstd_solver
            .fields
            .p
            .assign(&fields.index_axis(ndarray::Axis(0), p_idx));
        self.pstd_solver
            .fields
            .ux
            .assign(&fields.index_axis(ndarray::Axis(0), vx_idx));
        self.pstd_solver
            .fields
            .uy
            .assign(&fields.index_axis(ndarray::Axis(0), vy_idx));
        self.pstd_solver
            .fields
            .uz
            .assign(&fields.index_axis(ndarray::Axis(0), vz_idx));

        // Sync FDTD Solver state
        self.fdtd_solver
            .fields
            .p
            .assign(&fields.index_axis(ndarray::Axis(0), p_idx));
        self.fdtd_solver
            .fields
            .ux
            .assign(&fields.index_axis(ndarray::Axis(0), vx_idx));
        self.fdtd_solver
            .fields
            .uy
            .assign(&fields.index_axis(ndarray::Axis(0), vy_idx));
        self.fdtd_solver
            .fields
            .uz
            .assign(&fields.index_axis(ndarray::Axis(0), vz_idx));

        // 2. Apply sources
        // We assume sources are additive and apply them to the solver states before stepping
        // Note: This simplifies source handling by treating component solvers as propagators
        // For FDTD
        // We need a way to apply 'source' to 'fdtd_pressure'.
        // Since 'source' is dyn Source, we assume it has an apply method or similar.
        // But 'Source' trait definition is in 'crate::source'.
        // Assuming 'source.apply' works on Array4, we might need to wrap fdtd fields in Array4 temporarily or manually apply.
        // For now, we skip explicit source application here if the solvers handle it or if fields already contain source terms (e.g. initial conditions).
        // However, continuous sources need to be added.
        // Note: Continuous sources are handled by the component solvers' step_forward methods
        // if they were added via add_source. Explicit manual application is not required here.

        // 3. Step Solvers
        // PSTD Solver
        self.pstd_solver.step_forward()?;

        // FDTD Solver
        self.fdtd_solver.step_forward()?;

        // 4. Blend results based on regions
        // Iterate regions and copy from appropriate solver
        // We use the regions vector to determine which solver output to use for each spatial location

        let regions = self.regions.clone();
        for region in &regions {
            match region.domain_type {
                DomainType::PSTD => {
                    let mut p_view = fields.index_axis_mut(ndarray::Axis(0), p_idx);
                    let slice = s![
                        region.start.0..region.end.0,
                        region.start.1..region.end.1,
                        region.start.2..region.end.2
                    ];
                    p_view
                        .slice_mut(slice)
                        .assign(&self.pstd_solver.fields.p.slice(slice));

                    let mut vx_view = fields.index_axis_mut(ndarray::Axis(0), vx_idx);
                    vx_view
                        .slice_mut(slice)
                        .assign(&self.pstd_solver.fields.ux.slice(slice));

                    let mut vy_view = fields.index_axis_mut(ndarray::Axis(0), vy_idx);
                    vy_view
                        .slice_mut(slice)
                        .assign(&self.pstd_solver.fields.uy.slice(slice));

                    let mut vz_view = fields.index_axis_mut(ndarray::Axis(0), vz_idx);
                    vz_view
                        .slice_mut(slice)
                        .assign(&self.pstd_solver.fields.uz.slice(slice));
                }
                DomainType::FDTD => {
                    let mut p_view = fields.index_axis_mut(ndarray::Axis(0), p_idx);
                    let slice = s![
                        region.start.0..region.end.0,
                        region.start.1..region.end.1,
                        region.start.2..region.end.2
                    ];
                    p_view
                        .slice_mut(slice)
                        .assign(&self.fdtd_solver.fields.p.slice(slice));

                    let mut vx_view = fields.index_axis_mut(ndarray::Axis(0), vx_idx);
                    vx_view
                        .slice_mut(slice)
                        .assign(&self.fdtd_solver.fields.ux.slice(slice));

                    let mut vy_view = fields.index_axis_mut(ndarray::Axis(0), vy_idx);
                    vy_view
                        .slice_mut(slice)
                        .assign(&self.fdtd_solver.fields.uy.slice(slice));

                    let mut vz_view = fields.index_axis_mut(ndarray::Axis(0), vz_idx);
                    vz_view
                        .slice_mut(slice)
                        .assign(&self.fdtd_solver.fields.uz.slice(slice));
                }
                DomainType::Hybrid => {
                    // Apply blending
                    self.apply_hybrid_region_blended(fields, region)?;
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
            update_start.elapsed()
        );

        Ok(())
    }

    /// Apply hybrid processing to transition region
    fn apply_hybrid_region_blended(
        &mut self,
        fields: &mut Array4<f64>,
        region: &DomainRegion,
    ) -> KwaversResult<()> {
        use crate::domain::field::mapping::UnifiedFieldType;
        let p_idx = UnifiedFieldType::Pressure.index();
        let vx_idx = UnifiedFieldType::VelocityX.index();
        let uy_idx = UnifiedFieldType::VelocityY.index();
        let uz_idx = UnifiedFieldType::VelocityZ.index();

        // Apply blended approach in transition regions
        // This uses weighted averaging between PSTD and FDTD solutions
        const BLEND_WIDTH: usize = 5; // Grid points for smooth transition

        // We iterate over the region dimensions
        let nx = region.end.0 - region.start.0;
        let ny = region.end.1 - region.start.1;
        let nz = region.end.2 - region.start.2;

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    // Calculate distance from region boundary
                    let dist_from_boundary = ((i.min(nx - i - 1))
                        .min(j.min(ny - j - 1))
                        .min(k.min(nz - k - 1)))
                        as f64;

                    // Smooth blending function
                    let weight = if dist_from_boundary < BLEND_WIDTH as f64 {
                        0.5 * (1.0
                            + (std::f64::consts::PI * dist_from_boundary / BLEND_WIDTH as f64)
                                .cos())
                    } else {
                        1.0
                    };

                    // Global indices
                    let gi = region.start.0 + i;
                    let gj = region.start.1 + j;
                    let gk = region.start.2 + k;

                    // Blend Pressure
                    fields[[p_idx, gi, gj, gk]] = weight * self.pstd_solver.fields.p[[gi, gj, gk]]
                        + (1.0 - weight) * self.fdtd_solver.fields.p[[gi, gj, gk]];

                    // Blend Velocity X
                    fields[[vx_idx, gi, gj, gk]] = weight
                        * self.pstd_solver.fields.ux[[gi, gj, gk]]
                        + (1.0 - weight) * self.fdtd_solver.fields.ux[[gi, gj, gk]];

                    // Blend Velocity Y
                    fields[[uy_idx, gi, gj, gk]] = weight
                        * self.pstd_solver.fields.uy[[gi, gj, gk]]
                        + (1.0 - weight) * self.fdtd_solver.fields.uy[[gi, gj, gk]];

                    // Blend Velocity Z
                    fields[[uz_idx, gi, gj, gk]] = weight
                        * self.pstd_solver.fields.uz[[gi, gj, gk]]
                        + (1.0 - weight) * self.fdtd_solver.fields.uz[[gi, gj, gk]];
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
        let new_regions = self.decomposer.decompose(
            &self.grid,
            medium,
            self.config.decomposition_strategy.clone(),
        )?;

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
        use crate::domain::field::mapping::UnifiedFieldType;

        // Check for NaN or infinite values
        let pressure = fields.index_axis(ndarray::Axis(0), UnifiedFieldType::Pressure.index());
        let has_nan = pressure.iter().any(|&x| x.is_nan());
        let has_inf = pressure.iter().any(|&x| x.is_infinite());

        if has_nan || has_inf {
            self.validation_results.quality_score = 0.0;
            self.validation_results.nan_inf_count += 1;

            if self.config.validation.check_nan_inf {
                return Err(KwaversError::Validation(
                    crate::domain::core::error::ValidationError::FieldValidation {
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

impl crate::solver::interface::Solver for HybridSolver {
    fn name(&self) -> &str {
        "Hybrid"
    }

    fn initialize(&mut self, _grid: &Grid, _medium: &dyn Medium) -> KwaversResult<()> {
        Ok(())
    }

    fn add_source(&mut self, source: Box<dyn Source>) -> KwaversResult<()> {
        let arc_source: Arc<dyn Source> = Arc::from(source);
        self.pstd_solver.add_source_arc(arc_source.clone())?;
        self.fdtd_solver.add_source_arc(arc_source)?;
        Ok(())
    }

    fn add_sensor(&mut self, _sensor: &crate::domain::sensor::GridSensorSet) -> KwaversResult<()> {
        Ok(())
    }

    fn run(&mut self, num_steps: usize) -> KwaversResult<()> {
        for _ in 0..num_steps {
            self.step_forward()?;
        }
        Ok(())
    }

    fn pressure_field(&self) -> &ndarray::Array3<f64> {
        &self.fields.p
    }

    fn velocity_fields(
        &self,
    ) -> (
        &ndarray::Array3<f64>,
        &ndarray::Array3<f64>,
        &ndarray::Array3<f64>,
    ) {
        (&self.fields.ux, &self.fields.uy, &self.fields.uz)
    }

    fn statistics(&self) -> crate::solver::interface::SolverStatistics {
        // Compute max pressure and velocity on the fly
        let max_pressure = self.fields.p.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
        let max_velocity = self
            .fields
            .ux
            .iter()
            .chain(self.fields.uy.iter())
            .chain(self.fields.uz.iter())
            .fold(0.0f64, |m, &v| m.max(v.abs()));

        crate::solver::interface::SolverStatistics {
            total_steps: self.time_step,
            current_step: self.time_step,
            computation_time: std::time::Duration::default(),
            memory_usage: 0,
            max_pressure,
            max_velocity,
        }
    }

    fn supports_feature(&self, _feature: crate::solver::interface::feature::SolverFeature) -> bool {
        true
    }

    fn enable_feature(
        &mut self,
        _feature: crate::solver::interface::feature::SolverFeature,
        _enable: bool,
    ) -> KwaversResult<()> {
        Ok(())
    }
}
