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

    fn amplitude(&self, t: f64) -> f64 {
        self.signal.amplitude(t)
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        vec![self.position]
    }

    fn signal(&self) -> &dyn Signal {
        self.signal.as_ref()
    }
}

/// Time-varying source for time-reversal reconstruction
#[derive(Debug)]
pub struct TimeVaryingSource {
    position: (usize, usize, usize),
    signal_values: Vec<f64>,
    dt: f64,
    signal_wrapper: TimeVaryingSignal,
}

impl TimeVaryingSource {
    #[must_use]
    pub fn new(position: (usize, usize, usize), signal_values: Vec<f64>, dt: f64) -> Self {
        let signal_wrapper = TimeVaryingSignal::new(signal_values.clone(), dt);
        Self {
            position,
            signal_values,
            dt,
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

    fn amplitude(&self, t: f64) -> f64 {
        let index = (t / self.dt) as usize;
        if index < self.signal_values.len() {
            self.signal_values[index]
        } else {
            0.0
        }
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
        for source in &self.sources {
            let source_mask = source.create_mask(grid);
            mask = mask + source_mask; // Element-wise addition
        }
        mask
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

    fn amplitude(&self, _t: f64) -> f64 {
        0.0
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        vec![]
    }

    fn signal(&self) -> &dyn Signal {
        &self.null_signal
    }
}
