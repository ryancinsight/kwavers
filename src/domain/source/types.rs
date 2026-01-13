//! Source domain primitive
//!
//! Defines the core Source trait for wave generation.

// GridSource moved to parent module

use crate::domain::grid::Grid;
use crate::domain::signal::Signal;
use ndarray::Array3;
use std::fmt::Debug;

// GridSource re-exported by parent mod

use serde::{Deserialize, Serialize};

/// Type of source injection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SourceField {
    #[default]
    Pressure,
    VelocityX,
    VelocityY,
    VelocityZ,
}

/// Electromagnetic polarization state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Polarization {
    /// Linear polarization along x-axis
    LinearX,
    /// Linear polarization along y-axis
    LinearY,
    /// Linear polarization along z-axis
    LinearZ,
    /// Right circular polarization
    RightCircular,
    /// Left circular polarization
    LeftCircular,
    /// Elliptical polarization (ratio, phase difference)
    Elliptical { ratio: f64, phase_diff: f64 },
}

/// Electromagnetic wave type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EMWaveType {
    /// Transverse electromagnetic (no longitudinal components)
    TEM,
    /// Transverse electric (E_z = 0)
    TE,
    /// Transverse magnetic (H_z = 0)
    TM,
    /// Hybrid mode (both E_z, H_z nonzero)
    Hybrid,
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
