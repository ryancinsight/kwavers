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

// Keep MockSource for backward compatibility but mark as deprecated
#[deprecated(since = "1.0.0", note = "Use NullSource instead for testing")]
pub type MockSource = NullSource;

#[deprecated(since = "1.0.0", note = "Use NullSignal instead for testing")]
pub type MockSignal = NullSignal;
