// source/mod.rs

use crate::grid::Grid;
use crate::signal::Signal;
use std::fmt::Debug;

pub mod apodization;
pub mod linear_array;
pub mod matrix_array;
pub mod phased_array;

pub trait Source: Debug + Sync + Send {
    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, grid: &Grid) -> f64;
    fn positions(&self) -> Vec<(f64, f64, f64)>;
    fn signal(&self) -> &dyn Signal;
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
    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        // Simple point source - return signal amplitude at source position
        let dx = (x - self.position.0).abs();
        let dy = (y - self.position.1).abs();
        let dz = (z - self.position.2).abs();
        
        // Check if we're within one grid spacing of the source
        if dx < grid.dx && dy < grid.dy && dz < grid.dz {
            self.signal.amplitude(t)
        } else {
            0.0
        }
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
    
    fn get_current_amplitude(&self, t: f64) -> f64 {
        let index = (t / self.dt) as usize;
        if index < self.signal_values.len() {
            self.signal_values[index]
        } else {
            0.0
        }
    }
}

impl Source for TimeVaryingSource {
    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        // Check if this is the source position
        let (ix, iy, iz) = self.position;
        let (gx, gy, gz) = grid.position_to_indices(x, y, z);
        
        if ix == gx && iy == gy && iz == gz {
            self.get_current_amplitude(t)
        } else {
            0.0
        }
    }
    
    fn positions(&self) -> Vec<(f64, f64, f64)> {
        // Convert grid indices to physical positions
        vec![(
            self.position.0 as f64,
            self.position.1 as f64,
            self.position.2 as f64,
        )]
    }
    
    fn signal(&self) -> &dyn Signal {
        // Return a static signal since we can't create a temporary reference
        &NullSignal
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
    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        self.sources.iter()
            .map(|source| source.get_source_term(t, x, y, z, grid))
            .sum()
    }
    
    fn positions(&self) -> Vec<(f64, f64, f64)> {
        self.sources.iter()
            .flat_map(|source| source.positions())
            .collect()
    }
    
    fn signal(&self) -> &dyn Signal {
        // Return the first source's signal or null signal
        self.sources.first()
            .map(|s| s.signal())
            .unwrap_or(&NullSignal)
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

/// Null source implementation following the Null Object pattern
/// 
/// This provides a valid Source implementation that produces no acoustic output,
/// eliminating the need for null checks in client code.
#[derive(Debug, Clone)]
pub struct NullSource;

impl NullSource {
    pub fn new() -> Self {
        Self
    }
}

impl Default for NullSource {
    fn default() -> Self {
        Self::new()
    }
}

impl Source for NullSource {
    fn get_source_term(&self, _t: f64, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        0.0
    }
    
    fn signal(&self) -> &dyn Signal {
        &NullSignal
    }
    
    fn positions(&self) -> Vec<(f64, f64, f64)> {
        vec![]
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
