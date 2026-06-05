use super::HybridSolver;
use crate::forward::fdtd::FdtdSolver;
use crate::forward::hybrid::adaptive_selection::AdaptiveSelector;
use crate::forward::hybrid::config::HybridConfig;
use crate::forward::hybrid::coupling::CouplingInterface;
use crate::forward::hybrid::domain_decomposition::DomainDecomposer;
use crate::forward::hybrid::metrics::{HybridMetrics, HybridValidationResults};
use crate::forward::pstd::PSTDSolver;
use kwavers_core::error::KwaversResult;
use kwavers_field::wave::WaveFields;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use kwavers_source::GridSource;
use log::info;

impl HybridSolver {
    /// Create a new hybrid solver
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(config: HybridConfig, grid: &Grid, medium: &dyn Medium) -> KwaversResult<Self> {
        info!("Initializing hybrid Spectral/FDTD solver");

        let pstd_solver = PSTDSolver::new(
            config.pstd_config.clone(),
            grid.clone(),
            medium,
            GridSource::default(),
        )?;
        let fdtd_solver = FdtdSolver::new(
            config.fdtd_config.clone(),
            grid,
            medium,
            GridSource::default(),
        )?;

        let decomposer = DomainDecomposer::new();
        let selector = AdaptiveSelector::new(config.selection_criteria.clone());
        let coupling = CouplingInterface::new(
            grid,
            grid,
            crate::hybrid::coupling::HybridInterpolationScheme::Linear,
        )?;

        let default_medium = kwavers_medium::homogeneous::HomogeneousMedium::water(grid);
        let regions =
            decomposer.decompose(grid, &default_medium, config.decomposition_strategy.clone())?;

        info!("Hybrid solver initialized with {} regions", regions.len());

        let shape = (grid.nx, grid.ny, grid.nz);

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
            validation_results: HybridValidationResults::default(),
            time_step: 0,
            fields: WaveFields::new(shape),
            source_mask_scratch: ndarray::Array3::zeros(shape),
        })
    }
}
