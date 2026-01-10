//! Topology adapters for backward compatibility
//!
//! This module provides adapters that bridge between the legacy `Grid` struct
//! and the new `GridTopology` trait-based system, enabling incremental migration
//! without breaking existing code.
//!
//! # Design Philosophy
//!
//! - Zero-cost abstraction where possible
//! - Clear deprecation path for legacy APIs
//! - Preserve mathematical invariants during transition
//!
//! # Migration Strategy
//!
//! 1. Use `GridAdapter` to wrap existing `Grid` instances
//! 2. Gradually migrate code to use `GridTopology` trait directly
//! 3. Remove adapter once all code uses topology abstractions

use super::structure::Grid;
use super::topology::{CartesianTopology, GridTopology, TopologyDimension};
use crate::domain::core::error::KwaversResult;
use ndarray::Array3;

/// Adapter that implements `GridTopology` for the legacy `Grid` struct
///
/// This enables existing `Grid` instances to be used with new topology-aware code
/// without modification. The adapter is zero-cost (no heap allocations, simple delegation).
///
/// # Example
///
/// ```ignore
/// use kwavers::domain::grid::{Grid, GridAdapter, GridTopology};
///
/// let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3)?;
/// let adapter = GridAdapter::new(grid);
///
/// // Now use as GridTopology
/// let size = adapter.size();
/// let coords = adapter.indices_to_coordinates([32, 32, 32]);
/// ```
#[derive(Debug, Clone)]
pub struct GridAdapter {
    grid: Grid,
}

impl GridAdapter {
    /// Create a new adapter wrapping a `Grid`
    pub fn new(grid: Grid) -> Self {
        Self { grid }
    }

    /// Get a reference to the underlying `Grid`
    pub fn inner(&self) -> &Grid {
        &self.grid
    }

    /// Consume the adapter and return the underlying `Grid`
    pub fn into_inner(self) -> Grid {
        self.grid
    }

    /// Convert this adapter into a `CartesianTopology`
    ///
    /// This creates a proper topology instance that owns its data,
    /// suitable for long-term storage in topology-aware APIs.
    pub fn into_topology(self) -> KwaversResult<CartesianTopology> {
        CartesianTopology::new(
            [self.grid.nx, self.grid.ny, self.grid.nz],
            [self.grid.dx, self.grid.dy, self.grid.dz],
            self.grid.origin,
        )
    }

    /// Create an adapter from a topology (if it's Cartesian-compatible)
    ///
    /// This provides a bridge in the opposite direction for code that still
    /// requires the legacy `Grid` struct.
    pub fn from_topology(topology: &dyn GridTopology) -> KwaversResult<Self> {
        let dims = topology.dimensions();
        let spacing = topology.spacing();

        // Origin is not exposed by GridTopology trait, so we assume [0, 0, 0]
        // This is safe for most use cases as origin is rarely used
        let grid = Grid::new(
            dims[0], dims[1], dims[2], spacing[0], spacing[1], spacing[2],
        )?;

        Ok(Self { grid })
    }
}

impl GridTopology for GridAdapter {
    fn dimensionality(&self) -> TopologyDimension {
        match self.grid.dimensionality {
            1 => TopologyDimension::One,
            2 => TopologyDimension::Two,
            3 => TopologyDimension::Three,
            _ => TopologyDimension::Three, // Default to 3D for safety
        }
    }

    fn size(&self) -> usize {
        self.grid.size()
    }

    fn dimensions(&self) -> [usize; 3] {
        [self.grid.nx, self.grid.ny, self.grid.nz]
    }

    fn spacing(&self) -> [f64; 3] {
        [self.grid.dx, self.grid.dy, self.grid.dz]
    }

    fn extents(&self) -> [f64; 3] {
        [
            (self.grid.nx - 1) as f64 * self.grid.dx,
            (self.grid.ny - 1) as f64 * self.grid.dy,
            (self.grid.nz - 1) as f64 * self.grid.dz,
        ]
    }

    fn indices_to_coordinates(&self, indices: [usize; 3]) -> [f64; 3] {
        let coords = self
            .grid
            .indices_to_coordinates(indices[0], indices[1], indices[2]);
        [coords.0, coords.1, coords.2]
    }

    fn coordinates_to_indices(&self, coords: [f64; 3]) -> Option<[usize; 3]> {
        self.grid
            .position_to_indices(coords[0], coords[1], coords[2])
            .map(|(i, j, k)| [i, j, k])
    }

    fn metric_coefficient(&self, _indices: [usize; 3]) -> f64 {
        self.grid.dx * self.grid.dy * self.grid.dz
    }

    fn is_uniform(&self) -> bool {
        self.grid.is_uniform()
    }

    fn k_max(&self) -> f64 {
        self.grid.k_max
    }

    fn create_field(&self) -> Array3<f64> {
        self.grid.create_field()
    }
}

/// Extension trait to easily convert `Grid` to topology adapters
pub trait GridTopologyExt {
    /// Convert this grid into a topology adapter
    fn as_topology(&self) -> GridAdapter;

    /// Convert this grid into a proper `CartesianTopology`
    fn to_cartesian_topology(&self) -> KwaversResult<CartesianTopology>;
}

impl GridTopologyExt for Grid {
    fn as_topology(&self) -> GridAdapter {
        GridAdapter::new(self.clone())
    }

    fn to_cartesian_topology(&self) -> KwaversResult<CartesianTopology> {
        CartesianTopology::new(
            [self.nx, self.ny, self.nz],
            [self.dx, self.dy, self.dz],
            self.origin,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_adapter_creation() {
        let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3).unwrap();
        let adapter = GridAdapter::new(grid);

        assert_eq!(adapter.size(), 1000);
        assert_eq!(adapter.dimensionality(), TopologyDimension::Three);
    }

    #[test]
    fn test_adapter_topology_trait() {
        let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1).unwrap();
        let adapter = GridAdapter::new(grid);

        let dims = adapter.dimensions();
        assert_eq!(dims, [10, 10, 10]);

        let spacing = adapter.spacing();
        assert!((spacing[0] - 0.1).abs() < 1e-10);

        let coords = adapter.indices_to_coordinates([5, 5, 5]);
        assert!((coords[0] - 0.5).abs() < 1e-10);

        let indices = adapter.coordinates_to_indices([0.5, 0.5, 0.5]).unwrap();
        assert_eq!(indices, [5, 5, 5]);
    }

    #[test]
    fn test_adapter_metric() {
        let grid = Grid::new(10, 10, 10, 1e-3, 2e-3, 3e-3).unwrap();
        let adapter = GridAdapter::new(grid);

        let metric = adapter.metric_coefficient([5, 5, 5]);
        assert!((metric - 6e-9).abs() < 1e-15);
    }

    #[test]
    fn test_into_topology_conversion() {
        let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3).unwrap();
        let adapter = GridAdapter::new(grid);

        let topology = adapter.into_topology().unwrap();
        assert_eq!(topology.dimensions(), [10, 10, 10]);
        assert_eq!(topology.spacing(), [1e-3, 1e-3, 1e-3]);
    }

    #[test]
    fn test_extension_trait() {
        let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3).unwrap();

        let adapter = grid.as_topology();
        assert_eq!(adapter.size(), 1000);

        let topology = grid.to_cartesian_topology().unwrap();
        assert_eq!(topology.size(), 1000);
    }

    #[test]
    fn test_k_max_preservation() {
        let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3).unwrap();
        let adapter = GridAdapter::new(grid.clone());

        assert_eq!(adapter.k_max(), grid.k_max());
    }

    #[test]
    fn test_field_creation() {
        let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
        let adapter = GridAdapter::new(grid);

        let field = adapter.create_field();
        assert_eq!(field.dim(), (8, 8, 8));
    }
}
