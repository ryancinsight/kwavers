// physics/scattering/optic/mod.rs
pub mod rayleigh;

pub use rayleigh::RayleighOpticalScatteringModel;
use ndarray::Array3;
use crate::grid::Grid;
use crate::medium::Medium;
use std::fmt::Debug;
pub trait OpticalScatteringModel: Debug + Send + Sync {
    fn apply_scattering(&mut self, fluence: &mut Array3<f64>, grid: &Grid, medium: &dyn Medium);
}