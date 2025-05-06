// source/mod.rs

use crate::grid::Grid;
use crate::signal::Signal;
use std::fmt::Debug;

pub mod apodization;
pub mod focused_transducer;
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
pub use focused_transducer::FocusedTransducer;
pub use linear_array::LinearArray;
pub use matrix_array::MatrixArray;
