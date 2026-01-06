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
use crate::solver::spectral::{SpectralSolver, SpectralSource};
use crate::source::Source;
use log::{debug, info};
use ndarray::{s, Array3, Array4};
use std::time::Instant;

/// Context for regional solver application
#[allow(dead_code)]
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

    /// Spectral solver for smooth regions
    #[allow(dead_code)]
    spectral_solver: SpectralSolver,

    /// FDTD solver for discontinuous regions
    #[allow(dead_code)]
    fdtd_solver: FdtdSolver,

    /// FDTD scratch fields
    // Using Array4 to match fields structure or separate Array3s?
    // Actually simpler to store Array3s to match FdtdSolver signature
    fdtd_pressure: Array3<f64>,
    fdtd_vx: Array3<f64>,
    fdtd_vy: Array3<f64>,
    fdtd_vz: Array3<f64>,

    /// Material properties cache
    rho0: Array3<f64>,
    c0: Array3<f64>,

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
        let spectral_solver = SpectralSolver::new(
            config.spectral_config.clone(),
            grid.clone(),
            medium,
            SpectralSource::default(),
        )?;
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
        let regions =
            decomposer.decompose(grid, &default_medium, config.decomposition_strategy.clone())?;

        info!("Hybrid solver initialized with {} regions", regions.len());

        // Initialize material properties
        let mut rho0 = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut c0 = Array3::zeros((grid.nx, grid.ny, grid.nz));

        // Compute material properties
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                    rho0[[i, j, k]] = crate::medium::density_at(medium, x, y, z, grid);
                    c0[[i, j, k]] = crate::medium::sound_speed_at(medium, x, y, z, grid);
                }
            }
        }

        let shape = (grid.nx, grid.ny, grid.nz);

        Ok(Self {
            config,
            grid: grid.clone(),
            spectral_solver,
            fdtd_solver,
            fdtd_pressure: Array3::zeros(shape),
            fdtd_vx: Array3::zeros(shape),
            fdtd_vy: Array3::zeros(shape),
            fdtd_vz: Array3::zeros(shape),
            rho0,
            c0,
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

        // 1. Sync global fields to component solvers
        use crate::physics::field_mapping::UnifiedFieldType;
        let p_idx = UnifiedFieldType::Pressure.index();
        let vx_idx = UnifiedFieldType::VelocityX.index();
        let vy_idx = UnifiedFieldType::VelocityY.index();
        let vz_idx = UnifiedFieldType::VelocityZ.index();

        // Sync Spectral Solver
        self.spectral_solver
            .p
            .assign(&fields.index_axis(ndarray::Axis(0), p_idx));
        self.spectral_solver
            .ux
            .assign(&fields.index_axis(ndarray::Axis(0), vx_idx));
        self.spectral_solver
            .uy
            .assign(&fields.index_axis(ndarray::Axis(0), vy_idx));
        self.spectral_solver
            .uz
            .assign(&fields.index_axis(ndarray::Axis(0), vz_idx));

        // Sync FDTD Solver scratch
        self.fdtd_pressure
            .assign(&fields.index_axis(ndarray::Axis(0), p_idx));
        self.fdtd_vx
            .assign(&fields.index_axis(ndarray::Axis(0), vx_idx));
        self.fdtd_vy
            .assign(&fields.index_axis(ndarray::Axis(0), vy_idx));
        self.fdtd_vz
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
        // TODO: Implement proper source application logic.

        // 3. Step Solvers
        // Spectral Solver
        self.spectral_solver.step_forward()?;

        // FDTD Solver
        self.fdtd_solver.update_pressure(
            &mut self.fdtd_pressure,
            &self.fdtd_vx,
            &self.fdtd_vy,
            &self.fdtd_vz,
            self.rho0.view(),
            self.c0.view(),
            dt,
        )?;
        self.fdtd_solver.update_velocity(
            &mut self.fdtd_vx,
            &mut self.fdtd_vy,
            &mut self.fdtd_vz,
            &self.fdtd_pressure,
            self.rho0.view(),
            dt,
        )?;

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
                        .assign(&self.spectral_solver.p.slice(slice));

                    let mut vx_view = fields.index_axis_mut(ndarray::Axis(0), vx_idx);
                    vx_view
                        .slice_mut(slice)
                        .assign(&self.spectral_solver.ux.slice(slice));

                    let mut vy_view = fields.index_axis_mut(ndarray::Axis(0), vy_idx);
                    vy_view
                        .slice_mut(slice)
                        .assign(&self.spectral_solver.uy.slice(slice));

                    let mut vz_view = fields.index_axis_mut(ndarray::Axis(0), vz_idx);
                    vz_view
                        .slice_mut(slice)
                        .assign(&self.spectral_solver.uz.slice(slice));
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
                        .assign(&self.fdtd_pressure.slice(slice));

                    let mut vx_view = fields.index_axis_mut(ndarray::Axis(0), vx_idx);
                    vx_view.slice_mut(slice).assign(&self.fdtd_vx.slice(slice));

                    let mut vy_view = fields.index_axis_mut(ndarray::Axis(0), vy_idx);
                    vy_view.slice_mut(slice).assign(&self.fdtd_vy.slice(slice));

                    let mut vz_view = fields.index_axis_mut(ndarray::Axis(0), vz_idx);
                    vz_view.slice_mut(slice).assign(&self.fdtd_vz.slice(slice));
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
            start.elapsed()
        );

        Ok(())
    }

    /// Apply hybrid processing to transition region
    fn apply_hybrid_region_blended(
        &mut self,
        fields: &mut Array4<f64>,
        region: &DomainRegion,
    ) -> KwaversResult<()> {
        use crate::physics::field_mapping::UnifiedFieldType;
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
                    fields[[p_idx, gi, gj, gk]] = weight * self.spectral_solver.p[[gi, gj, gk]]
                        + (1.0 - weight) * self.fdtd_pressure[[gi, gj, gk]];

                    // Blend Velocity X
                    fields[[vx_idx, gi, gj, gk]] = weight * self.spectral_solver.ux[[gi, gj, gk]]
                        + (1.0 - weight) * self.fdtd_vx[[gi, gj, gk]];

                    // Blend Velocity Y
                    fields[[uy_idx, gi, gj, gk]] = weight * self.spectral_solver.uy[[gi, gj, gk]]
                        + (1.0 - weight) * self.fdtd_vy[[gi, gj, gk]];

                    // Blend Velocity Z
                    fields[[uz_idx, gi, gj, gk]] = weight * self.spectral_solver.uz[[gi, gj, gk]]
                        + (1.0 - weight) * self.fdtd_vz[[gi, gj, gk]];
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
