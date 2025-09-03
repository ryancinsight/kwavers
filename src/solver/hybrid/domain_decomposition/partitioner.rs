//! Domain partitioning algorithms

use super::analyzer::QualityMetrics;
use super::region::{DomainRegion, DomainType};
use crate::error::KwaversResult;
use crate::grid::Grid;

/// Partitions domain into regions for different solvers
#[derive(Debug)]
pub struct DomainPartitioner {
    /// Minimum region size to avoid excessive fragmentation
    min_region_size: usize,
    /// Score threshold for PSTD selection
    pstd_threshold: f64,
    /// Score threshold for FDTD selection
    fdtd_threshold: f64,
}

impl DomainPartitioner {
    /// Create a new domain partitioner
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_region_size: 8,  // Minimum 8x8x8 region
            pstd_threshold: 0.7, // High homogeneity/smoothness for PSTD
            fdtd_threshold: 0.3, // Low homogeneity/smoothness for FDTD
        }
    }

    /// Partition the domain based on quality metrics
    pub fn partition(
        &self,
        grid: &Grid,
        metrics: &QualityMetrics,
    ) -> KwaversResult<Vec<DomainRegion>> {
        let mut regions = Vec::new();

        // Simple partitioning: divide into uniform blocks and classify each
        // In production, use adaptive partitioning algorithms
        let block_size = 32.max(self.min_region_size);

        for i in (0..grid.nx).step_by(block_size) {
            for j in (0..grid.ny).step_by(block_size) {
                for k in (0..grid.nz).step_by(block_size) {
                    let start = (i, j, k);
                    let end = (
                        (i + block_size).min(grid.nx),
                        (j + block_size).min(grid.ny),
                        (k + block_size).min(grid.nz),
                    );

                    let score = self.compute_region_score(start, end, metrics);
                    let domain_type = self.classify_region(score);

                    regions.push(DomainRegion::new(start, end, domain_type, score));
                }
            }
        }

        Ok(regions)
    }

    /// Compute quality score for a region
    fn compute_region_score(
        &self,
        start: (usize, usize, usize),
        end: (usize, usize, usize),
        metrics: &QualityMetrics,
    ) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;

        for i in start.0..end.0 {
            for j in start.1..end.1 {
                for k in start.2..end.2 {
                    if i < metrics.homogeneity.dim().0
                        && j < metrics.homogeneity.dim().1
                        && k < metrics.homogeneity.dim().2
                    {
                        sum += metrics.homogeneity[[i, j, k]] * 0.4
                            + metrics.smoothness[[i, j, k]] * 0.4
                            + metrics.spectral_content[[i, j, k]] * 0.2;
                        count += 1;
                    }
                }
            }
        }

        if count > 0 {
            sum / f64::from(count)
        } else {
            0.5
        }
    }

    /// Classify region based on score
    fn classify_region(&self, score: f64) -> DomainType {
        if score >= self.pstd_threshold {
            DomainType::PSTD
        } else if score <= self.fdtd_threshold {
            DomainType::FDTD
        } else {
            DomainType::Hybrid
        }
    }
}

impl Default for DomainPartitioner {
    fn default() -> Self {
        Self::new()
    }
}
