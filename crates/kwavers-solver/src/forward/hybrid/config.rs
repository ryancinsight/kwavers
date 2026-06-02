//! Configuration structures for hybrid PSTD/FDTD solver

use crate::forward::fdtd::FdtdConfig;
use crate::forward::hybrid::adaptive_selection::HybridSelectionCriteria;
use crate::forward::hybrid::domain_decomposition::DomainRegion;
use crate::forward::pstd::PSTDConfig;
use serde::{Deserialize, Serialize};

/// Domain decomposition strategy
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HybridDecompositionStrategy {
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
    pub decomposition_strategy: HybridDecompositionStrategy,

    /// Adaptive selection parameters
    pub selection_criteria: HybridSelectionCriteria,

    /// Coupling interface configuration
    pub coupling_interface: CouplingInterfaceConfig,

    /// Performance optimization settings
    pub optimization: HybridSolverOptimizationConfig,

    /// Validation settings
    pub validation: HybridValidationConfig,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            pstd_config: PSTDConfig::default(),
            fdtd_config: FdtdConfig::default(),
            decomposition_strategy: HybridDecompositionStrategy::Dynamic,
            selection_criteria: HybridSelectionCriteria::default(),
            coupling_interface: CouplingInterfaceConfig::default(),
            optimization: HybridSolverOptimizationConfig::default(),
            validation: HybridValidationConfig::default(),
        }
    }
}

/// Configuration for coupling interface between PSTD and FDTD regions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouplingInterfaceConfig {
    /// Interpolation scheme for data exchange
    pub interpolation_scheme: HybridConfigInterpolationScheme,

    /// Number of ghost cells for coupling
    pub ghost_cells: usize,

    /// Enable filtering at interfaces
    pub enable_filtering: bool,
}

impl Default for CouplingInterfaceConfig {
    fn default() -> Self {
        Self {
            interpolation_scheme: HybridConfigInterpolationScheme::Cubic,
            ghost_cells: 4,
            enable_filtering: true,
        }
    }
}

/// Interpolation scheme for coupling interfaces
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum HybridConfigInterpolationScheme {
    /// Linear interpolation
    Linear,
    /// Cubic interpolation
    Cubic,
    /// Spectral interpolation
    Spectral,
}

/// Performance optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSolverOptimizationConfig {
    /// Enable cache optimization
    pub cache_optimization: bool,

    /// Enable SIMD vectorization
    pub simd_enabled: bool,

    /// Thread pool size (0 for auto)
    pub thread_pool_size: usize,
}

impl Default for HybridSolverOptimizationConfig {
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
pub struct HybridValidationConfig {
    /// Enable solution validation
    pub enable_validation: bool,

    /// Check for NaN/Inf values
    pub check_nan_inf: bool,

    /// Maximum allowed relative error
    pub max_relative_error: f64,
}

impl Default for HybridValidationConfig {
    fn default() -> Self {
        Self {
            enable_validation: true,
            check_nan_inf: true,
            max_relative_error: 1e-3,
        }
    }
}
