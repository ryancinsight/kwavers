// boundary/mod.rs

use crate::grid::Grid;
use ndarray::Array3;
use std::fmt::Debug;

pub mod pml;

pub trait Boundary: Debug + Send + Sync {
    /// Applies boundary conditions to the acoustic pressure field in spatial domain.
    fn apply_acoustic(&self, field: &mut Array3<f64>, grid: &Grid, time_step: usize);

    /// Applies boundary conditions to the acoustic field in frequency domain (k-space).
    fn apply_acoustic_freq(
        &self,
        field: &mut Array3<rustfft::num_complex::Complex<f64>>,
        grid: &Grid,
        time_step: usize,
    );

    /// Applies boundary conditions to the light fluence rate field (spatial domain).
    fn apply_light(&self, field: &mut Array3<f64>, grid: &Grid, time_step: usize);
}

pub use pml::PMLBoundary;
