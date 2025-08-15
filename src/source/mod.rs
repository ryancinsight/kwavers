// source/mod.rs

use crate::grid::Grid;
use crate::signal::Signal;
use ndarray::Array3;
use std::fmt::Debug;

pub mod apodization;
pub mod flexible_transducer;
pub mod focused_transducer;
pub mod hemispherical_array;
pub mod linear_array;
pub mod matrix_array;
pub mod phased_array;

/// Efficient source trait using mask-based approach
pub trait Source: Debug + Sync + Send {
    /// Create a source mask on the grid (1.0 at source locations, 0.0 elsewhere)
    /// This is called once during initialization for optimal performance
    fn create_mask(&self, grid: &Grid) -> Array3<f64>;
    
    /// Get the signal amplitude at time t
    /// This is called once per time step, not per grid point
    fn amplitude(&self, t: f64) -> f64;
    
    /// Get source positions for visualization/analysis
    fn positions(&self) -> Vec<(f64, f64, f64)>;
    
    /// Get the underlying signal
    fn signal(&self) -> &dyn Signal;
    
    /// Legacy method for backward compatibility - DEPRECATED
    /// Use create_mask() and amplitude() for better performance
    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        // Fallback implementation - inefficient but maintains compatibility
        if let Some((i, j, k)) = grid.to_grid_indices(x, y, z) {
            let mask = self.create_mask(grid);
            mask.get((i, j, k)).copied().unwrap_or(0.0) * self.amplitude(t)
        } else {
            0.0
        }
    }
}

pub use apodization::{
    Apodization, BlackmanApodization, GaussianApodization, HammingApodization, HanningApodization,
    RectangularApodization,
};
pub use linear_array::LinearArray;
pub use matrix_array::MatrixArray;
pub use phased_array::{
    PhasedArrayTransducer, PhasedArrayConfig, TransducerElement, 
    ElementSensitivity, BeamformingMode
};
pub use focused_transducer::{
    BowlTransducer, BowlConfig, ArcSource, ArcConfig,
    MultiBowlArray, ApodizationType, make_bowl, make_annular_array
};

// Source implementations are defined in this module
// No need to re-export them from self

/// Simple point source implementation
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
        if let Some((ix, iy, iz)) = grid.to_grid_indices(self.position.0, self.position.1, self.position.2) {
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
}

impl TimeVaryingSource {
    pub fn new(position: (usize, usize, usize), signal_values: Vec<f64>, dt: f64) -> Self {
        Self {
            position,
            signal_values,
            dt,
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
        // Convert grid indices to physical coordinates
        let x = self.position.0 as f64; // Simplified - would need grid spacing
        let y = self.position.1 as f64;
        let z = self.position.2 as f64;
        vec![(x, y, z)]
    }
    
    fn signal(&self) -> &dyn Signal {
        &NullSignal // Placeholder since this uses direct signal values
    }
}

/// Composite source that combines multiple sources
#[derive(Debug)]
pub struct CompositeSource {
    sources: Vec<Box<dyn Source>>,
}

impl CompositeSource {
    pub fn new(sources: Vec<Box<dyn Source>>) -> Self {
        Self { sources }
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
        self.sources.iter()
            .map(|source| source.amplitude(t))
            .sum()
    }
    
    fn positions(&self) -> Vec<(f64, f64, f64)> {
        self.sources.iter()
            .flat_map(|source| source.positions())
            .collect()
    }
    
    fn signal(&self) -> &dyn Signal {
        // Return the first source's signal as a representative
        if let Some(first_source) = self.sources.first() {
            first_source.signal()
        } else {
            &NullSignal
        }
    }
}

/// Null source implementation for testing
#[derive(Debug)]
pub struct NullSource;

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
        &NullSignal
    }
}

/// Signal implementation for time-varying source
#[derive(Debug)]
struct TimeVaryingSignal<'a> {
    values: &'a [f64],
    dt: f64,
}

impl<'a> Signal for TimeVaryingSignal<'a> {
    fn amplitude(&self, t: f64) -> f64 {
        let index = (t / self.dt) as usize;
        if index < self.values.len() {
            self.values[index]
        } else {
            0.0
        }
    }
    
    fn frequency(&self, _t: f64) -> f64 {
        // Time-varying signal doesn't have a single frequency
        0.0
    }
    
    fn phase(&self, _t: f64) -> f64 {
        0.0
    }
    
    fn clone_box(&self) -> Box<dyn Signal> {
        Box::new(NullSignal)
    }
}

/// Null signal implementation
#[derive(Debug, Clone)]
struct NullSignal;

impl Signal for NullSignal {
    fn amplitude(&self, _t: f64) -> f64 {
        0.0
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
