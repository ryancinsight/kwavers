//! Result types for GPU SWE operations.

use kwavers_grid::Grid;

/// GPU propagation result
#[derive(Debug, Clone)]
pub struct GPUPropagationResult {
    /// Total execution time (seconds)
    pub execution_time: f64,
    /// Kernel execution time (seconds)
    pub kernel_time: f64,
    /// Memory used (bytes)
    pub memory_used: usize,
    /// Computational throughput (cells/second)
    pub throughput: f64,
    /// GPU grid dimensions
    pub grid_size: [usize; 3],
    /// GPU block dimensions
    pub block_size: [usize; 3],
}

/// GPU inversion result
#[derive(Debug, Clone)]
pub struct GPUInversionResult {
    /// Total execution time (seconds)
    pub execution_time: f64,
    /// Kernel execution time (seconds)
    pub kernel_time: f64,
    /// Memory used (bytes)
    pub memory_used: usize,
    /// Number of directions processed
    pub directions_processed: usize,
    /// Number of convergence iterations
    pub convergence_iterations: usize,
    /// Final residual error
    pub residual_error: f64,
}

/// Adaptive solution result
#[derive(Debug, Clone)]
pub struct AdaptiveSolution {
    /// Solution steps at different resolutions
    pub steps: Vec<AdaptiveSolutionStep>,
    /// Final solution quality (0-1)
    pub final_quality: f64,
    /// Total computation time (seconds)
    pub total_computation_time: f64,
}

/// Single adaptive solution step
#[derive(Debug, Clone)]
pub struct AdaptiveSolutionStep {
    /// Resolution level
    pub level: usize,
    /// Grid at this resolution
    pub grid: Grid,
    /// Solution quality (0-1)
    pub quality: f64,
    /// Computation time (seconds)
    pub computation_time: f64,
}
