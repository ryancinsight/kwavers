//! Configuration structures for hybrid PSTD/FDTD solver

use crate::solver::forward::fdtd::FdtdConfig;
use crate::solver::forward::hybrid::adaptive_selection::SelectionCriteria;
use crate::solver::forward::hybrid::domain_decomposition::DomainRegion;
use crate::solver::forward::pstd::PSTDConfig;
use serde::{Deserialize, Serialize};

/// Domain decomposition strategy
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DecompositionStrategy {
    /// Static decomposition based on initial conditions
    Static,
    /// Dynamic decomposition that adapts during simulation
    Dynamic,
    /// User-defined regions
    UserDefined(Vec<DomainRegion>),
    /// Frequency-based decomposition
    FrequencyBased,
}

/// Configuration for the hybrid Spectral/FDTD solver
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridConfig {
    /// Spectral solver configuration
    pub pstd_config: PSTDConfig,

    /// FDTD solver configuration
    pub fdtd_config: FdtdConfig,

    /// Domain decomposition strategy
    pub decomposition_strategy: DecompositionStrategy,

    /// Adaptive selection parameters
    pub selection_criteria: SelectionCriteria,

    /// Coupling interface configuration
    pub coupling_interface: CouplingInterfaceConfig,

    /// Performance optimization settings
    pub optimization: OptimizationConfig,

    /// Validation settings
    pub validation: ValidationConfig,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            pstd_config: PSTDConfig::default(),
            fdtd_config: FdtdConfig::default(),
            decomposition_strategy: DecompositionStrategy::Dynamic,
            selection_criteria: SelectionCriteria::default(),
            coupling_interface: CouplingInterfaceConfig::default(),
            optimization: OptimizationConfig::default(),
            validation: ValidationConfig::default(),
        }
    }
}

/// Configuration for coupling interface between PSTD and FDTD regions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouplingInterfaceConfig {
    /// Interpolation scheme for data exchange
    pub interpolation_scheme: InterpolationScheme,

    /// Number of ghost cells for coupling
    pub ghost_cells: usize,

    /// Enable filtering at interfaces
    pub enable_filtering: bool,
}

impl Default for CouplingInterfaceConfig {
    fn default() -> Self {
        Self {
            interpolation_scheme: InterpolationScheme::Cubic,
            ghost_cells: 4,
            enable_filtering: true,
        }
    }
}

/// Interpolation scheme for coupling interfaces
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum InterpolationScheme {
    /// Linear interpolation
    Linear,
    /// Cubic interpolation
    Cubic,
    /// Spectral interpolation
    Spectral,
}

/// Performance optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Enable cache optimization
    pub cache_optimization: bool,

    /// Enable SIMD vectorization
    pub simd_enabled: bool,

    /// Thread pool size (0 for auto)
    pub thread_pool_size: usize,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            cache_optimization: true,
            simd_enabled: true,
            thread_pool_size: 0,
        }
    }
}

/// Validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Enable solution validation
    pub enable_validation: bool,

    /// Check for NaN/Inf values
    pub check_nan_inf: bool,

    /// Maximum allowed relative error
    pub max_relative_error: f64,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enable_validation: true,
            check_nan_inf: true,
            max_relative_error: 1e-3,
        }
    }
}
