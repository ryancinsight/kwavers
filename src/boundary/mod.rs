// boundary/mod.rs

use crate::KwaversResult;
use crate::grid::Grid;
use ndarray::Array3;
use std::fmt::Debug;

pub mod pml;

/// Trait for boundary condition implementations
/// Follows Interface Segregation Principle - clients depend only on methods they use
pub trait Boundary: Debug + Send + Sync {
    /// Applies boundary conditions to the acoustic field in spatial domain.
    ///
    /// # Arguments
    ///
    /// * `field` - The acoustic pressure field to apply boundary conditions to
    /// * `grid` - The simulation grid
    /// * `time_step` - Current simulation time step
    fn apply_acoustic(&mut self, field: &mut Array3<f64>, grid: &Grid, time_step: usize) -> KwaversResult<()>;

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
    ) -> KwaversResult<()>;

    /// Applies boundary conditions to the acoustic field with a scaling factor.
    ///
    /// # Arguments
    ///
    /// * `field` - The acoustic pressure field to apply boundary conditions to
    /// * `grid` - The simulation grid
    /// * `time_step` - Current simulation time step
    /// * `factor` - Scaling factor for boundary application
    fn apply_acoustic_with_factor(&mut self, field: &mut Array3<f64>, grid: &Grid, time_step: usize, _factor: f64) -> KwaversResult<()> {
        // Default implementation applies regular boundary conditions
        self.apply_acoustic(field, grid, time_step)?;
        // Scale the boundary effects by the factor if needed
        // This is a default implementation - specific boundary types can override
        Ok(())
    }

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
