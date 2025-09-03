//! Domain region definitions for hybrid solver decomposition

use serde::{Deserialize, Serialize};

/// Type of solver optimal for a domain region
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DomainType {
    /// Pseudospectral method optimal
    PSTD,
    /// Finite-difference time-domain optimal
    FDTD,
    /// Hybrid approach needed
    Hybrid,
}

/// Domain region with associated solver type
#[derive(Debug, Clone)]
pub struct DomainRegion {
    /// Starting indices (inclusive)
    pub start: (usize, usize, usize),
    /// Ending indices (exclusive)
    pub end: (usize, usize, usize),
    /// Optimal solver type for this region
    pub domain_type: DomainType,
    /// Quality score for this assignment
    pub quality_score: f64,
}

impl DomainRegion {
    /// Create a new domain region
    #[must_use]
    pub fn new(
        start: (usize, usize, usize),
        end: (usize, usize, usize),
        domain_type: DomainType,
        quality_score: f64,
    ) -> Self {
        Self {
            start,
            end,
            domain_type,
            quality_score,
        }
    }

    /// Get the size of this region in each dimension
    #[must_use]
    pub fn size(&self) -> (usize, usize, usize) {
        (
            self.end.0 - self.start.0,
            self.end.1 - self.start.1,
            self.end.2 - self.start.2,
        )
    }

    /// Get the total number of grid points in this region
    #[must_use]
    pub fn volume(&self) -> usize {
        let (nx, ny, nz) = self.size();
        nx * ny * nz
    }

    /// Check if a point is within this region
    #[must_use]
    pub fn contains(&self, i: usize, j: usize, k: usize) -> bool {
        i >= self.start.0
            && i < self.end.0
            && j >= self.start.1
            && j < self.end.1
            && k >= self.start.2
            && k < self.end.2
    }
}
