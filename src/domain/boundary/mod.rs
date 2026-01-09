// boundary/mod.rs
//
// Runtime boundary conditions live under `simulation::environment`.
// This module must not depend on crate-root re-exports; use direct imports only.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::{Array3, ArrayViewMut3};
use std::fmt::Debug;

pub mod config;
pub mod cpml;
pub mod pml;

pub use config::{BoundaryParameters, BoundaryType};

/// Trait for runtime boundary condition implementations.
///
/// ## Multi-physics note
/// Kwavers supports optics in addition to acoustics. The runtime boundary trait
/// therefore includes `apply_light(...)` for fluence/fluence-rate boundary handling.
pub trait Boundary: Debug + Send + Sync {
    /// Applies boundary conditions to the acoustic field in spatial domain.
    fn apply_acoustic(
        &mut self,
        field: ArrayViewMut3<f64>,
        grid: &Grid,
        time_step: usize,
    ) -> KwaversResult<()>;

    /// Applies boundary conditions to the acoustic field in frequency domain (k-space).
    fn apply_acoustic_freq(
        &mut self,
        field: &mut Array3<rustfft::num_complex::Complex<f64>>,
        grid: &Grid,
        time_step: usize,
    ) -> KwaversResult<()>;

    /// Applies boundary conditions to the acoustic field with a scaling factor.
    ///
    /// Default implementation applies regular boundary conditions.
    fn apply_acoustic_with_factor(
        &mut self,
        field: ArrayViewMut3<f64>,
        grid: &Grid,
        time_step: usize,
        _factor: f64,
    ) -> KwaversResult<()> {
        self.apply_acoustic(field, grid, time_step)
    }

    /// Applies boundary conditions to the light fluence rate field (spatial domain).
    fn apply_light(&mut self, field: ArrayViewMut3<f64>, grid: &Grid, time_step: usize);
}

pub use cpml::{CPMLBoundary, CPMLConfig};
pub use pml::{PMLBoundary, PMLConfig};
