//! Multi-region construction and sampling failures.

use kwavers_grid::geometry::GeometryError;
use thiserror::Error;

/// Failure to construct or sample a multi-region PINN domain.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Error)]
pub enum MultiRegionError {
    /// At least one region is required.
    #[error("a multi-region domain requires at least one region")]
    Empty,
    /// Every region requires one material identifier.
    #[error("material count mismatch: {regions} regions, {materials} material identifiers")]
    MaterialCount {
        /// Region count.
        regions: usize,
        /// Material identifier count.
        materials: usize,
    },
    /// `N` regions require `N-1` interface conditions.
    #[error("interface count mismatch: expected {expected} for {regions} regions, got {actual}")]
    InterfaceCount {
        /// Region count.
        regions: usize,
        /// Required interface count.
        expected: usize,
        /// Supplied interface count.
        actual: usize,
    },
    /// All regions in one interface matrix must have the same dimension.
    #[error("region {region} dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Invalid region index.
        region: usize,
        /// First-region dimension.
        expected: usize,
        /// Invalid region dimension.
        actual: usize,
    },
    /// Underlying geometric sampling failed.
    #[error(transparent)]
    Geometry(#[from] GeometryError),
}
