// source/mod.rs

use crate::grid::Grid;
use crate::signal::{NullSignal, Signal, TimeVaryingSignal};
use ndarray::Array3;
use std::fmt::Debug;
use std::sync::Arc;

// Hierarchical source module structure
pub mod basic;
pub mod custom;
pub mod transducers;
pub mod wavefront;

/// Type of source injection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SourceField {
    #[default]
    Pressure,
    VelocityX,
    VelocityY,
    VelocityZ,
}

pub type SourceType = SourceField;

/// Efficient source trait using mask-based approach
pub trait Source: Debug + Sync + Send {
    /// Create a source mask on the grid (1.0 at source locations, 0.0 elsewhere)
    /// This is called once during initialization for optimal performance
    fn create_mask(&self, grid: &Grid) -> Array3<f64>;

    /// Get the signal amplitude at time t
    /// This is called once per time step, not per grid point
    fn amplitude(&self, t: f64) -> f64;

    /// Get source positions for visualization/analysis
    fn positions(&self) -> Vec<(f64, f64, f64)>;

    /// Get the underlying signal
    fn signal(&self) -> &dyn Signal;

    /// Get the type of source
    fn source_type(&self) -> SourceField {
        SourceField::Pressure
    }

    /// Get the initial amplitude (for p0/u0)
    /// If non-zero, this source contributes to the initial conditions
    fn initial_amplitude(&self) -> f64 {
        0.0
    }

    /// Get source term at a specific position and time
    /// Uses `create_mask()` and `amplitude()` internally for compatibility
    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        // Fallback implementation - inefficient but maintains compatibility
        if let Some((i, j, k)) = grid.position_to_indices(x, y, z) {
            let mask = self.create_mask(grid);
            mask.get((i, j, k)).copied().unwrap_or(0.0) * self.amplitude(t)
        } else {
            0.0
        }
    }
}

// Re-export all source types for easy access
pub use basic::{
    linear_array::LinearArray,
    matrix_array::MatrixArray,
    piston::{PistonApodization, PistonBuilder, PistonConfig, PistonSource},
};
pub use custom::{
    CustomSourceBuilder, FunctionSource, SimpleCustomSource, SimpleCustomSourceBuilder,
};
pub use transducers::{
    apodization::{
        Apodization, BlackmanApodization, GaussianApodization, HammingApodization,
        HanningApodization, RectangularApodization,
    },
    focused::{
        make_annular_array, make_bowl, ApodizationType, ArcConfig, ArcSource, BowlConfig,
        BowlTransducer, MultiBowlArray,
    },
    phased_array::{
        BeamformingMode, ElementSensitivity, PhasedArrayConfig, PhasedArrayTransducer,
        TransducerElement,
    },
};
pub use wavefront::{
    bessel::{BesselBuilder, BesselConfig, BesselSource},
    gaussian::{GaussianBuilder, GaussianConfig, GaussianSource},
    plane_wave::{PlaneWaveBuilder, PlaneWaveConfig, PlaneWaveSource},
    spherical::{SphericalBuilder, SphericalConfig, SphericalSource, SphericalWaveType},
};

/// Point source implementation
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
    fn create_mask(&self, grid: &Grid) -> Array3<f64> {
        let mut mask = Array3::zeros((grid.nx, grid.ny, grid.nz));
        if let Some((ix, iy, iz)) =
            grid.position_to_indices(self.position.0, self.position.1, self.position.2)
        {
            mask[(ix, iy, iz)] = 1.0;
        }
        mask
    }

    fn amplitude(&self, t: f64) -> f64 {
        self.signal.amplitude(t)
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
    signal_wrapper: TimeVaryingSignal,
}

impl TimeVaryingSource {
    #[must_use]
    pub fn new(position: (usize, usize, usize), signal_values: Vec<f64>, dt: f64) -> Self {
        let signal_wrapper =
            crate::signal::special::TimeVaryingSignal::new(signal_values.clone(), dt);
        Self {
            position,
            signal_values,
            dt,
            signal_wrapper,
        }
    }
}

impl Source for TimeVaryingSource {
    fn create_mask(&self, grid: &Grid) -> Array3<f64> {
        let mut mask = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let (ix, iy, iz) = self.position;
        if ix < grid.nx && iy < grid.ny && iz < grid.nz {
            mask[(ix, iy, iz)] = 1.0;
        }
        mask
    }

    fn amplitude(&self, t: f64) -> f64 {
        let index = (t / self.dt) as usize;
        if index < self.signal_values.len() {
            self.signal_values[index]
        } else {
            0.0
        }
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        // Return grid indices as positions - caller should convert using grid spacing
        // This maintains consistency with the mask which uses grid indices
        let x = self.position.0 as f64;
        let y = self.position.1 as f64;
        let z = self.position.2 as f64;
        vec![(x, y, z)]
    }

    fn signal(&self) -> &dyn Signal {
        &self.signal_wrapper
    }
}

/// Composite source that combines multiple sources
#[derive(Debug)]
pub struct CompositeSource {
    sources: Vec<Box<dyn Source>>,
    null_signal: NullSignal,
}

impl CompositeSource {
    #[must_use]
    pub fn new(sources: Vec<Box<dyn Source>>) -> Self {
        Self {
            sources,
            null_signal: NullSignal,
        }
    }
}

impl Source for CompositeSource {
    fn create_mask(&self, grid: &Grid) -> Array3<f64> {
        let mut mask = Array3::zeros((grid.nx, grid.ny, grid.nz));
        for source in &self.sources {
            let source_mask = source.create_mask(grid);
            mask = mask + source_mask; // Element-wise addition
        }
        mask
    }

    fn amplitude(&self, t: f64) -> f64 {
        self.sources.iter().map(|source| source.amplitude(t)).sum()
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        self.sources
            .iter()
            .flat_map(|source| source.positions())
            .collect()
    }

    fn signal(&self) -> &dyn Signal {
        // Return the first source's signal as a representative
        if let Some(first_source) = self.sources.first() {
            first_source.signal()
        } else {
            &self.null_signal
        }
    }
}

/// Null source implementation for testing
#[derive(Debug)]
pub struct NullSource {
    null_signal: NullSignal,
}

impl NullSource {
    #[must_use]
    pub fn new() -> Self {
        Self {
            null_signal: NullSignal::new(),
        }
    }
}

impl Default for NullSource {
    fn default() -> Self {
        Self::new()
    }
}

impl Source for NullSource {
    fn create_mask(&self, grid: &Grid) -> Array3<f64> {
        Array3::zeros((grid.nx, grid.ny, grid.nz))
    }

    fn amplitude(&self, _t: f64) -> f64 {
        0.0
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        vec![]
    }

    fn signal(&self) -> &dyn Signal {
        &self.null_signal
    }
}
