//! Domain Decomposition for Hybrid PSTD/FDTD Solver
//!
//! This module implements intelligent domain decomposition algorithms that analyze
//! the computational domain and automatically partition it into regions where
//! either PSTD, FDTD, or hybrid methods provide optimal performance.
//!
//! # Physics-Based Selection Criteria:
//!
//! ## PSTD Optimal Regions:
//! - **Homogeneous media**: Constant material properties
//! - **Smooth fields**: Low spatial gradients, high spectral content
//! - **Far-field regions**: Distance from sources and boundaries
//! - **High-frequency dominance**: Where spectral accuracy is critical
//!
//! ## FDTD Optimal Regions:
//! - **Heterogeneous media**: Material interfaces and discontinuities
//! - **Complex geometries**: Curved boundaries, fine structures
//! - **Near-field regions**: Close to sources and scatterers
//! - **Shock formation**: Steep gradients and nonlinear effects
//!
//! ## Hybrid Regions:
//! - **Transition zones**: Gradual changes in material properties
//! - **Intermediate smoothness**: Neither fully smooth nor highly discontinuous
//! - **Multi-scale features**: Mixed frequency content

pub mod analyzer;
pub mod buffer;
pub mod metrics;
pub mod partitioner;
pub mod region;

pub use analyzer::DomainAnalyzer;
pub use buffer::{BufferZones, OverlapRegion};
pub use metrics::QualityMetrics;
pub use partitioner::DomainPartitioner;
pub use region::{DomainRegion, DomainType};

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;

/// Main domain decomposer that coordinates analysis and partitioning
#[derive(Debug, Debug))]
pub struct DomainDecomposer {
    analyzer: DomainAnalyzer,
    partitioner: DomainPartitioner,
}

impl DomainDecomposer {
    /// Create a new domain decomposer with default parameters
    pub fn new() -> Self {
        Self {
            analyzer: DomainAnalyzer::new(),
            partitioner: DomainPartitioner::new(),
        }
    }

    /// Decompose the domain based on medium properties and grid
    pub fn decompose(&self, grid: &Grid, medium: &dyn Medium) -> KwaversResult<Vec<DomainRegion>> {
        // Analyze the domain
        let metrics = self.analyzer.analyze(grid, medium)?;

        // Partition based on analysis
        self.partitioner.partition(grid, &metrics)
    }
}

impl Default for DomainDecomposer {
    fn default() -> Self {
        Self::new()
    }
}
