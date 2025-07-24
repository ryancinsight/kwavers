// source/mod.rs

use crate::grid::Grid;
use crate::signal::Signal;
use std::fmt::Debug;

pub mod apodization;
pub mod linear_array;
pub mod matrix_array;

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

/// Mock signal implementation for testing purposes
#[derive(Debug, Clone)]
pub struct MockSignal {
    amplitude: f64,
    frequency: f64,
}

impl MockSignal {
    pub fn new(amplitude: f64, frequency: f64) -> Self {
        Self { amplitude, frequency }
    }
}

impl Default for MockSignal {
    fn default() -> Self {
        Self::new(1.0, 1e6) // 1.0 amplitude, 1 MHz default
    }
}

impl Signal for MockSignal {
    fn amplitude(&self, t: f64) -> f64 {
        if t >= 0.0 && t <= 1e-6 {
            // 1 μs pulse duration
            self.amplitude * (2.0 * std::f64::consts::PI * self.frequency * t).sin()
        } else {
            0.0
        }
    }

    fn frequency(&self, _t: f64) -> f64 {
        self.frequency
    }

    fn duration(&self) -> Option<f64> {
        Some(1e-6) // 1 μs duration
    }
    
    fn phase(&self, t: f64) -> f64 {
        2.0 * std::f64::consts::PI * self.frequency * t
    }
    
    fn clone_box(&self) -> Box<dyn Signal> {
        Box::new(self.clone())
    }
}

/// Mock source implementation for testing and development
#[derive(Debug)]
pub struct MockSource {
    signal: MockSignal,
    positions: Vec<(f64, f64, f64)>,
}

impl MockSource {
    pub fn new() -> Self {
        Self {
            signal: MockSignal::default(),
            positions: vec![(0.0, 0.0, 0.0)], // Single source at origin
        }
    }
    
    /// Create a mock source with custom signal parameters
    pub fn with_signal(amplitude: f64, frequency: f64) -> Self {
        Self {
            signal: MockSignal::new(amplitude, frequency),
            positions: vec![(0.0, 0.0, 0.0)],
        }
    }
    
    /// Create a mock source with custom positions
    pub fn with_positions(positions: Vec<(f64, f64, f64)>) -> Self {
        Self {
            signal: MockSignal::default(),
            positions,
        }
    }
    
    /// Create a fully customized mock source
    pub fn custom(amplitude: f64, frequency: f64, positions: Vec<(f64, f64, f64)>) -> Self {
        Self {
            signal: MockSignal::new(amplitude, frequency),
            positions,
        }
    }
}

impl Default for MockSource {
    fn default() -> Self {
        Self::new()
    }
}

impl Source for MockSource {
    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, _grid: &Grid) -> f64 {
        // Calculate source term for all positions
        let mut total_source = 0.0;
        
        for &(sx, sy, sz) in &self.positions {
            let distance = ((x - sx).powi(2) + (y - sy).powi(2) + (z - sz).powi(2)).sqrt();
            
            // Gaussian spatial distribution with 1mm characteristic width
            let spatial_factor = (-0.5 * (distance / 1e-3).powi(2)).exp();
            
            // Time-dependent signal
            let temporal_factor = self.signal.amplitude(t);
            
            total_source += spatial_factor * temporal_factor;
        }
        
        total_source
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        self.positions.clone()
    }

    fn signal(&self) -> &dyn Signal {
        &self.signal
    }
}
