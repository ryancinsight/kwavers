//! Custom source types
//!
//! This module provides interfaces for creating custom source implementations
//! that can be defined by users for specialized applications.

use crate::{Source, SourceField};
use kwavers_grid::Grid;
use kwavers_signal::Signal;
use ndarray::Array3;
use std::fmt::Debug;
use std::sync::Arc;

/// Custom source builder trait
/// Allows users to create custom sources by implementing this trait
pub trait CustomSourceBuilder: Debug + Sync + Send {
    /// Build the custom source
    fn build(&self, signal: Arc<dyn Signal>) -> Box<dyn Source>;
}

/// Simple custom source implementation
/// Users can extend this for their specific needs
#[derive(Debug)]
pub struct SimpleCustomSource {
    positions: Vec<(f64, f64, f64)>,
    weights: Vec<f64>,
    signal: Arc<dyn Signal>,
    source_type: SourceField,
}

impl SimpleCustomSource {
    /// Create a new simple custom source
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    pub fn new(
        positions: Vec<(f64, f64, f64)>,
        weights: Vec<f64>,
        signal: Arc<dyn Signal>,
        source_type: SourceField,
    ) -> Self {
        assert_eq!(
            positions.len(),
            weights.len(),
            "Positions and weights must have the same length"
        );
        Self {
            positions,
            weights,
            signal,
            source_type,
        }
    }

    /// Create a custom source builder
    #[must_use]
    pub fn builder() -> SimpleCustomSourceBuilder {
        SimpleCustomSourceBuilder::new()
    }
}

impl Source for SimpleCustomSource {
    fn create_mask(&self, grid: &Grid) -> Array3<f64> {
        let mut mask = Array3::zeros((grid.nx, grid.ny, grid.nz));
        self.create_mask_into(grid, &mut mask);
        mask
    }

    fn create_mask_into(&self, grid: &Grid, mask: &mut Array3<f64>) {
        debug_assert_eq!(mask.dim(), (grid.nx, grid.ny, grid.nz));
        mask.fill(0.0);

        for (pos_idx, (x, y, z)) in self.positions.iter().enumerate() {
            if let Some((i, j, k)) = grid.position_to_indices(*x, *y, *z) {
                mask[(i, j, k)] = self.weights[pos_idx];
            }
        }
    }

    fn amplitude(&self, t: f64) -> f64 {
        self.signal.amplitude(t)
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        self.positions.clone()
    }

    fn signal(&self) -> &dyn Signal {
        self.signal.as_ref()
    }

    fn source_type(&self) -> SourceField {
        self.source_type
    }

    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let Some(query_index) = grid.position_to_indices(x, y, z) else {
            return 0.0;
        };

        let mut weight_at_cell = None;
        for (pos_idx, (px, py, pz)) in self.positions.iter().enumerate() {
            if grid.position_to_indices(*px, *py, *pz) == Some(query_index) {
                weight_at_cell = Some(self.weights[pos_idx]);
            }
        }

        weight_at_cell.map_or(0.0, |weight| weight * self.signal.amplitude(t))
    }
}

/// Builder for simple custom sources
#[derive(Debug, Default)]
pub struct SimpleCustomSourceBuilder {
    positions: Vec<(f64, f64, f64)>,
    weights: Vec<f64>,
    source_type: SourceField,
}

impl SimpleCustomSourceBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn add_position(mut self, position: (f64, f64, f64), weight: f64) -> Self {
        self.positions.push(position);
        self.weights.push(weight);
        self
    }

    #[must_use]
    pub fn source_type(mut self, source_type: SourceField) -> Self {
        self.source_type = source_type;
        self
    }

    pub fn build(self, signal: Arc<dyn Signal>) -> SimpleCustomSource {
        SimpleCustomSource::new(self.positions, self.weights, signal, self.source_type)
    }
}

impl CustomSourceBuilder for SimpleCustomSourceBuilder {
    fn build(&self, signal: Arc<dyn Signal>) -> Box<dyn Source> {
        Box::new(SimpleCustomSource::new(
            self.positions.clone(),
            self.weights.clone(),
            signal,
            self.source_type,
        ))
    }
}

/// Function-based custom source
/// Allows users to define sources using arbitrary functions
pub struct FunctionSource {
    function: Box<dyn Fn(f64, f64, f64, f64) -> f64 + Sync + Send>,
    signal: Arc<dyn Signal>,
    source_type: SourceField,
}

impl std::fmt::Debug for FunctionSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionSource")
            .field("source_type", &self.source_type)
            .field("signal", &"<function>")
            .field("function", &"<closure>")
            .finish()
    }
}

impl FunctionSource {
    /// Create a new function-based source
    pub fn new<F: Fn(f64, f64, f64, f64) -> f64 + Sync + Send + 'static>(
        function: F,
        signal: Arc<dyn Signal>,
        source_type: SourceField,
    ) -> Self {
        Self {
            function: Box::new(function),
            signal,
            source_type,
        }
    }
}

impl Source for FunctionSource {
    fn create_mask(&self, grid: &Grid) -> Array3<f64> {
        let mut mask = Array3::zeros((grid.nx, grid.ny, grid.nz));
        self.create_mask_into(grid, &mut mask);
        mask
    }

    fn create_mask_into(&self, grid: &Grid, mask: &mut Array3<f64>) {
        debug_assert_eq!(mask.dim(), (grid.nx, grid.ny, grid.nz));

        for ((i, j, k), val) in mask.indexed_iter_mut() {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k as f64 * grid.dz;

            // Evaluate the function at time 0 to get the spatial distribution
            *val = (self.function)(x, y, z, 0.0);
        }
    }

    fn add_mask_into(&self, grid: &Grid, mask: &mut Array3<f64>) {
        debug_assert_eq!(mask.dim(), (grid.nx, grid.ny, grid.nz));

        for ((i, j, k), val) in mask.indexed_iter_mut() {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k as f64 * grid.dz;

            *val += (self.function)(x, y, z, 0.0);
        }
    }

    fn amplitude(&self, t: f64) -> f64 {
        self.signal.amplitude(t)
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        // For function sources, we can't easily determine positions
        // Return empty vector
        vec![]
    }

    fn signal(&self) -> &dyn Signal {
        self.signal.as_ref()
    }

    fn source_type(&self) -> SourceField {
        self.source_type
    }

    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, _grid: &Grid) -> f64 {
        (self.function)(x, y, z, t) * self.signal.amplitude(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct ConstantSignal(f64);

    impl Signal for ConstantSignal {
        fn amplitude(&self, _t: f64) -> f64 {
            self.0
        }

        fn frequency(&self, _t: f64) -> f64 {
            0.0
        }

        fn phase(&self, _t: f64) -> f64 {
            0.0
        }

        fn clone_box(&self) -> Box<dyn Signal> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn simple_custom_source_term_matches_mask_cells_not_nearest_position() {
        let grid = Grid::new(4, 4, 4, 1.0, 1.0, 1.0).unwrap();
        let source = SimpleCustomSource::new(
            vec![(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)],
            vec![2.0, 7.0],
            Arc::new(ConstantSignal(3.0)),
            SourceField::Pressure,
        );
        let mask = source.create_mask(&grid);
        let source_cell = grid.indices_to_coordinates(1, 1, 1);
        let empty_cell = grid.indices_to_coordinates(2, 1, 1);

        assert_eq!(mask[[1, 1, 1]], 7.0);
        assert_eq!(
            source.get_source_term(0.0, source_cell.0, source_cell.1, source_cell.2, &grid),
            21.0
        );
        assert_eq!(
            source.get_source_term(0.0, empty_cell.0, empty_cell.1, empty_cell.2, &grid),
            0.0
        );
    }
}
