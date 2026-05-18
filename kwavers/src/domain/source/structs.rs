use crate::domain::grid::Grid;
use crate::domain::signal::{NullSignal, Signal, TimeVaryingSignal};
use crate::domain::source::types::Source;
use ndarray::Array3;
use std::fmt::Debug;

/// Point source implementation
#[derive(Debug)]
pub struct PointSource {
    position: (f64, f64, f64),
    signal: std::sync::Arc<dyn Signal>,
}

impl PointSource {
    pub fn new(position: (f64, f64, f64), signal: std::sync::Arc<dyn Signal>) -> Self {
        Self { position, signal }
    }
}

impl Source for PointSource {
    fn create_mask(&self, grid: &Grid) -> Array3<f64> {
        let mut mask = Array3::zeros((grid.nx, grid.ny, grid.nz));
        if let Some((ix, iy, iz)) =
            grid.position_to_indices(self.position.0, self.position.1, self.position.2)
        {
            mask[(ix, iy, iz)] = 1.0;
        }
        mask
    }

    fn add_mask_into(&self, grid: &Grid, mask: &mut Array3<f64>) {
        debug_assert_eq!(mask.dim(), (grid.nx, grid.ny, grid.nz));
        if let Some((ix, iy, iz)) =
            grid.position_to_indices(self.position.0, self.position.1, self.position.2)
        {
            mask[(ix, iy, iz)] += 1.0;
        }
    }

    fn amplitude(&self, t: f64) -> f64 {
        self.signal.amplitude(t)
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        vec![self.position]
    }

    fn signal(&self) -> &dyn Signal {
        self.signal.as_ref()
    }

    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let source_index =
            grid.position_to_indices(self.position.0, self.position.1, self.position.2);
        if source_index.is_some() && grid.position_to_indices(x, y, z) == source_index {
            self.signal.amplitude(t)
        } else {
            0.0
        }
    }
}

/// Time-varying source for time-reversal reconstruction
#[derive(Debug)]
pub struct TimeVaryingSource {
    position: (usize, usize, usize),
    signal_wrapper: TimeVaryingSignal,
}

impl TimeVaryingSource {
    #[must_use]
    pub fn new(position: (usize, usize, usize), signal_values: Vec<f64>, dt: f64) -> Self {
        let signal_wrapper = TimeVaryingSignal::new(signal_values, dt);
        Self {
            position,
            signal_wrapper,
        }
    }
}

impl Source for TimeVaryingSource {
    fn create_mask(&self, grid: &Grid) -> Array3<f64> {
        let mut mask = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let (ix, iy, iz) = self.position;
        if ix < grid.nx && iy < grid.ny && iz < grid.nz {
            mask[(ix, iy, iz)] = 1.0;
        }
        mask
    }

    fn add_mask_into(&self, grid: &Grid, mask: &mut Array3<f64>) {
        debug_assert_eq!(mask.dim(), (grid.nx, grid.ny, grid.nz));
        let (ix, iy, iz) = self.position;
        if ix < grid.nx && iy < grid.ny && iz < grid.nz {
            mask[(ix, iy, iz)] += 1.0;
        }
    }

    fn amplitude(&self, t: f64) -> f64 {
        self.signal_wrapper.amplitude(t)
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        let x = self.position.0 as f64;
        let y = self.position.1 as f64;
        let z = self.position.2 as f64;
        vec![(x, y, z)]
    }

    fn signal(&self) -> &dyn Signal {
        &self.signal_wrapper
    }

    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        if grid.position_to_indices(x, y, z) == Some(self.position) {
            self.amplitude(t)
        } else {
            0.0
        }
    }
}

/// Composite source that combines multiple sources
#[derive(Debug)]
pub struct CompositeSource {
    sources: Vec<Box<dyn Source>>,
    null_signal: NullSignal,
}

impl CompositeSource {
    #[must_use]
    pub fn new(sources: Vec<Box<dyn Source>>) -> Self {
        Self {
            sources,
            null_signal: NullSignal::new(),
        }
    }
}

impl Source for CompositeSource {
    fn create_mask(&self, grid: &Grid) -> Array3<f64> {
        let mut mask = Array3::zeros((grid.nx, grid.ny, grid.nz));
        self.create_mask_into(grid, &mut mask);
        mask
    }

    fn add_mask_into(&self, grid: &Grid, mask: &mut Array3<f64>) {
        debug_assert_eq!(mask.dim(), (grid.nx, grid.ny, grid.nz));
        for source in &self.sources {
            source.add_mask_into(grid, mask);
        }
    }

    fn create_mask_into(&self, grid: &Grid, mask: &mut Array3<f64>) {
        debug_assert_eq!(mask.dim(), (grid.nx, grid.ny, grid.nz));
        mask.fill(0.0);
        self.add_mask_into(grid, mask);
    }

    fn amplitude(&self, t: f64) -> f64 {
        self.sources.iter().map(|source| source.amplitude(t)).sum()
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        self.sources
            .iter()
            .flat_map(|source| source.positions())
            .collect()
    }

    fn signal(&self) -> &dyn Signal {
        if let Some(first_source) = self.sources.first() {
            first_source.signal()
        } else {
            &self.null_signal
        }
    }

    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        self.sources
            .iter()
            .map(|source| source.get_source_term(t, x, y, z, grid))
            .sum()
    }
}

/// Null source implementation for testing
#[derive(Debug)]
pub struct NullSource {
    null_signal: NullSignal,
}

impl NullSource {
    #[must_use]
    pub fn new() -> Self {
        Self {
            null_signal: NullSignal::new(),
        }
    }
}

impl Default for NullSource {
    fn default() -> Self {
        Self::new()
    }
}

impl Source for NullSource {
    fn create_mask(&self, grid: &Grid) -> Array3<f64> {
        Array3::zeros((grid.nx, grid.ny, grid.nz))
    }

    fn create_mask_into(&self, grid: &Grid, mask: &mut Array3<f64>) {
        debug_assert_eq!(mask.dim(), (grid.nx, grid.ny, grid.nz));
        mask.fill(0.0);
    }

    fn add_mask_into(&self, grid: &Grid, mask: &mut Array3<f64>) {
        debug_assert_eq!(mask.dim(), (grid.nx, grid.ny, grid.nz));
    }

    fn amplitude(&self, _t: f64) -> f64 {
        0.0
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        vec![]
    }

    fn signal(&self) -> &dyn Signal {
        &self.null_signal
    }

    fn get_source_term(&self, _t: f64, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

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
    fn point_source_mask_into_matches_owned_mask_without_reallocation() {
        let grid = Grid::new(4, 4, 4, 1.0, 1.0, 1.0).unwrap();
        let signal = Arc::new(NullSignal::new());
        let source = PointSource::new((1.0, 2.0, 3.0), signal);
        let owned = source.create_mask(&grid);
        let mut reused = Array3::from_elem((grid.nx, grid.ny, grid.nz), 9.0);
        let ptr = reused.as_ptr();

        source.create_mask_into(&grid, &mut reused);

        assert_eq!(reused, owned);
        assert_eq!(reused.as_ptr(), ptr);
    }

    #[test]
    fn composite_source_mask_into_adds_child_masks_without_reallocation() {
        let grid = Grid::new(4, 4, 4, 1.0, 1.0, 1.0).unwrap();
        let signal = Arc::new(NullSignal::new());
        let source = CompositeSource::new(vec![
            Box::new(PointSource::new((1.0, 1.0, 1.0), signal.clone())),
            Box::new(PointSource::new((1.0, 1.0, 1.0), signal)),
            Box::new(NullSource::new()),
        ]);
        let owned = source.create_mask(&grid);
        let mut reused = Array3::from_elem((grid.nx, grid.ny, grid.nz), -1.0);
        let ptr = reused.as_ptr();

        source.create_mask_into(&grid, &mut reused);

        assert_eq!(reused, owned);
        assert_eq!(reused[[1, 1, 1]], 2.0);
        assert_eq!(reused.as_ptr(), ptr);
    }

    #[test]
    fn point_source_term_is_nonzero_only_on_source_cell() {
        let grid = Grid::new(4, 4, 4, 1.0, 1.0, 1.0).unwrap();
        let source = PointSource::new((1.0, 2.0, 3.0), Arc::new(ConstantSignal(3.5)));
        let source_cell = grid.indices_to_coordinates(1, 2, 3);
        let empty_cell = grid.indices_to_coordinates(1, 2, 2);

        assert_eq!(
            source.get_source_term(0.25, source_cell.0, source_cell.1, source_cell.2, &grid),
            3.5
        );
        assert_eq!(
            source.get_source_term(0.25, empty_cell.0, empty_cell.1, empty_cell.2, &grid),
            0.0
        );
    }

    #[test]
    fn time_varying_source_term_uses_indexed_signal_on_source_cell() {
        let grid = Grid::new(4, 4, 4, 1.0, 1.0, 1.0).unwrap();
        let source = TimeVaryingSource::new((2, 1, 0), vec![1.0, 2.0, 4.0], 0.5);
        let source_cell = grid.indices_to_coordinates(2, 1, 0);
        let empty_cell = grid.indices_to_coordinates(2, 1, 1);

        assert_eq!(
            source.get_source_term(1.0, source_cell.0, source_cell.1, source_cell.2, &grid),
            4.0
        );
        assert_eq!(
            source.get_source_term(1.0, empty_cell.0, empty_cell.1, empty_cell.2, &grid),
            0.0
        );
    }

    #[test]
    fn composite_source_term_sums_local_children_without_cross_bleed() {
        let grid = Grid::new(4, 4, 4, 1.0, 1.0, 1.0).unwrap();
        let source = CompositeSource::new(vec![
            Box::new(PointSource::new(
                (1.0, 1.0, 1.0),
                Arc::new(ConstantSignal(2.0)),
            )),
            Box::new(PointSource::new(
                (2.0, 1.0, 1.0),
                Arc::new(ConstantSignal(5.0)),
            )),
        ]);
        let first_cell = grid.indices_to_coordinates(1, 1, 1);
        let second_cell = grid.indices_to_coordinates(2, 1, 1);
        let empty_cell = grid.indices_to_coordinates(3, 1, 1);

        assert_eq!(
            source.get_source_term(0.0, first_cell.0, first_cell.1, first_cell.2, &grid),
            2.0
        );
        assert_eq!(
            source.get_source_term(0.0, second_cell.0, second_cell.1, second_cell.2, &grid),
            5.0
        );
        assert_eq!(
            source.get_source_term(0.0, empty_cell.0, empty_cell.1, empty_cell.2, &grid),
            0.0
        );
    }

    #[test]
    fn null_source_term_is_identically_zero() {
        let grid = Grid::new(2, 2, 2, 1.0, 1.0, 1.0).unwrap();
        let source = NullSource::new();

        assert_eq!(source.get_source_term(0.0, 0.0, 0.0, 0.0, &grid), 0.0);
        assert_eq!(source.get_source_term(1.0, 1.0, 1.0, 1.0, &grid), 0.0);
    }
}
