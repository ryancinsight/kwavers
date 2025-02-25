// boundary/mod.rs

use crate::grid::Grid;
use ndarray::Array3;
use std::fmt::Debug;

pub mod pml;

/// Trait for implementing boundary conditions in the simulation.
///
/// Boundary conditions are applied to both acoustic and light fields
/// in both spatial and frequency domains.
pub trait Boundary: Debug + Send + Sync {
    /// Applies boundary conditions to the acoustic pressure field in spatial domain.
    ///
    /// # Arguments
    ///
    /// * `field` - The acoustic pressure field to apply boundary conditions to
    /// * `grid` - The simulation grid
    /// * `time_step` - Current simulation time step
    fn apply_acoustic(&mut self, field: &mut Array3<f64>, grid: &Grid, time_step: usize);

    /// Applies boundary conditions to the acoustic field in frequency domain (k-space).
    ///
    /// # Arguments
    ///
    /// * `field` - The complex acoustic field in frequency domain
    /// * `grid` - The simulation grid
    /// * `time_step` - Current simulation time step
    fn apply_acoustic_freq(
        &mut self,
        field: &mut Array3<rustfft::num_complex::Complex<f64>>,
        grid: &Grid,
        time_step: usize,
    );

    /// Applies boundary conditions to the light fluence rate field (spatial domain).
    ///
    /// # Arguments
    ///
    /// * `field` - The light fluence rate field to apply boundary conditions to
    /// * `grid` - The simulation grid
    /// * `time_step` - Current simulation time step
    fn apply_light(&mut self, field: &mut Array3<f64>, grid: &Grid, time_step: usize);
}

pub use pml::PMLBoundary;
