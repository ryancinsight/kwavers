//! Core hybrid PSTD/FDTD solver implementation

use crate::domain::field::wave::WaveFields;
use crate::domain::grid::Grid;
use crate::solver::forward::fdtd::FdtdSolver;
use crate::solver::forward::hybrid::adaptive_selection::AdaptiveSelector;
use crate::solver::forward::hybrid::config::HybridConfig;
use crate::solver::forward::hybrid::coupling::CouplingInterface;
use crate::solver::forward::hybrid::domain_decomposition::{DomainDecomposer, DomainRegion};
use crate::solver::forward::hybrid::metrics::{HybridMetrics, ValidationResults};
use crate::solver::forward::pstd::PSTDSolver;

mod construction;
mod interface_impl;
mod stepping;
mod update;

/// Hybrid PSTD/FDTD solver combining spectral and finite-difference methods
#[derive(Debug)]
pub struct HybridSolver {
    /// Configuration
    pub(super) config: HybridConfig,

    /// Computational grid
    pub(super) grid: Grid,

    /// PSTD solver for smooth regions
    pub(super) pstd_solver: PSTDSolver,

    /// FDTD solver for discontinuous regions
    pub(super) fdtd_solver: FdtdSolver,

    // Unified Fields
    pub(super) fields: WaveFields,

    /// Domain decomposer
    pub(super) decomposer: DomainDecomposer,

    /// Adaptive selector for method choice
    pub(super) selector: AdaptiveSelector,

    /// Coupling interface manager
    pub(super) coupling: CouplingInterface,

    /// Current domain regions
    pub(super) regions: Vec<DomainRegion>,

    /// Performance metrics
    pub(super) metrics: HybridMetrics,

    /// Validation results
    pub(super) validation_results: ValidationResults,

    /// Time step counter
    pub(super) time_step: usize,
}
