//! Performance and validation metrics for hybrid solver

use std::collections::HashMap;
use std::time::Duration;

/// Performance metrics for hybrid solver
#[derive(Debug, Clone, Default)]
pub struct HybridMetrics {
    /// Time spent in PSTD regions
    pub pstd_time: Duration,

    /// Time spent in FDTD regions
    pub fdtd_time: Duration,

    /// Time spent in coupling interfaces
    pub coupling_time: Duration,

    /// Time spent in domain decomposition
    pub decomposition_time: Duration,

    /// Number of PSTD grid points
    pub pstd_points: usize,

    /// Number of FDTD grid points
    pub fdtd_points: usize,

    /// Method-specific metrics
    pub method_metrics: HashMap<String, f64>,
}

impl HybridMetrics {
    /// Create new metrics instance
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate total computation time
    #[must_use]
    pub fn total_time(&self) -> Duration {
        self.pstd_time + self.fdtd_time + self.coupling_time + self.decomposition_time
    }

    /// Calculate PSTD fraction of computation
    #[must_use]
    pub fn pstd_fraction(&self) -> f64 {
        let total = (self.pstd_points + self.fdtd_points) as f64;
        if total > 0.0 {
            self.pstd_points as f64 / total
        } else {
            0.0
        }
    }
}

/// Efficiency metrics for method selection
#[derive(Debug, Clone, Default)]
pub struct EfficiencyMetrics {
    /// Computational efficiency score (0-1)
    pub efficiency_score: f64,

    /// Memory bandwidth utilization (0-1)
    pub bandwidth_utilization: f64,

    /// Cache hit rate (0-1)
    pub cache_hit_rate: f64,
}

/// Validation results for solution quality
#[derive(Debug, Clone, Default)]
pub struct ValidationResults {
    /// Solution quality score (0-1)
    pub quality_score: f64,

    /// Maximum relative error
    pub max_relative_error: f64,

    /// Number of NaN/Inf values detected
    pub nan_inf_count: usize,
}
