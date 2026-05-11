use super::HybridSolver;
use crate::core::error::KwaversResult;
use crate::domain::field::wave::WaveFields;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::source::GridSource;
use crate::solver::forward::fdtd::FdtdSolver;
use crate::solver::forward::hybrid::adaptive_selection::AdaptiveSelector;
use crate::solver::forward::hybrid::config::HybridConfig;
use crate::solver::forward::hybrid::coupling::CouplingInterface;
use crate::solver::forward::hybrid::domain_decomposition::DomainDecomposer;
use crate::solver::forward::hybrid::metrics::{HybridMetrics, ValidationResults};
use crate::solver::forward::pstd::PSTDSolver;
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
            crate::solver::hybrid::coupling::InterpolationScheme::Linear,
        )?;

        let default_medium = crate::domain::medium::homogeneous::HomogeneousMedium::water(grid);
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
            validation_results: ValidationResults::default(),
            time_step: 0,
            fields: WaveFields::new(shape),
        })
    }
}
