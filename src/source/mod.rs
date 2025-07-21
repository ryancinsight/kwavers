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

#[cfg(test)]
#[derive(Debug)]
pub struct MockSource {}

#[cfg(test)]
impl MockSource {
    pub fn new() -> Self {
        Self {}
    }
}

#[cfg(test)]
impl Default for MockSource {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
impl Source for MockSource {
    fn get_source_term(&self, _t: f64, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        0.0
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        vec![]
    }

    fn signal(&self) -> &dyn Signal {
        // This is tricky because we need a concrete type that implements Signal.
        // For now, we can panic, or create a mock signal if needed.
        panic!("MockSource::signal() should not be called in this test")
    }
}
