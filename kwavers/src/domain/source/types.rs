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

/// Focal properties for focused wave sources
///
/// These properties characterize the focusing behavior of sources like
/// focused transducers, Gaussian beams, and phased arrays.
#[derive(Debug, Clone, Copy)]
pub struct FocalProperties {
    /// Focal point position (m)
    pub focal_point: (f64, f64, f64),

    /// Focal depth/length: distance from source center to focal point (m)
    pub focal_depth: f64,

    /// Spot size at focus: beam waist or FWHM (m)
    pub spot_size: f64,

    /// F-number: focal_length / aperture_diameter (dimensionless)
    pub f_number: Option<f64>,

    /// Rayleigh range: depth of focus (m)
    pub rayleigh_range: Option<f64>,

    /// Numerical aperture: sin(half_angle) (dimensionless)
    pub numerical_aperture: Option<f64>,

    /// Focal gain: intensity amplification at focus (dimensionless)
    pub focal_gain: Option<f64>,
}

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

    // ==================== Focal Properties API ====================
    // These methods enable PINN adapters and analysis tools to extract
    // focal properties from sources without tight coupling to concrete types.

    /// Get focal point position (m)
    ///
    /// Returns the position where the wave field converges to maximum intensity.
    /// Returns `None` for unfocused sources (plane waves, point sources, etc.)
    fn focal_point(&self) -> Option<(f64, f64, f64)> {
        None
    }

    /// Get focal depth (m)
    ///
    /// Distance from the source center/aperture to the focal point.
    /// Also known as focal length for geometric focusing.
    fn focal_depth(&self) -> Option<f64> {
        None
    }

    /// Get spot size at focus (m)
    ///
    /// Minimum transverse beam dimension at the focal point.
    /// - For Gaussian beams: beam waist w0
    /// - For focused transducers: FWHM of main lobe
    /// - For diffraction-limited systems: ~λ * f_number
    fn spot_size(&self) -> Option<f64> {
        None
    }

    /// Get F-number (dimensionless)
    ///
    /// Ratio of focal length to aperture diameter: F = f / D
    /// Characterizes the focusing strength and depth of field:
    /// - Small F# (<1): Strong focusing, shallow depth
    /// - Large F# (>3): Weak focusing, large depth
    fn f_number(&self) -> Option<f64> {
        None
    }

    /// Get Rayleigh range (m)
    ///
    /// Depth of focus: distance over which beam radius ≤ √2 * spot_size.
    /// For Gaussian beams: z_R = π * w0² / λ
    /// For focused transducers: approximately λ * F#²
    fn rayleigh_range(&self) -> Option<f64> {
        None
    }

    /// Get numerical aperture (dimensionless)
    ///
    /// NA = sin(θ) where θ is the half-angle of the convergence cone.
    /// Related to F-number by: NA ≈ 1 / (2 * F#)
    /// Higher NA → stronger focusing, better resolution
    fn numerical_aperture(&self) -> Option<f64> {
        None
    }

    /// Get focal gain (dimensionless)
    ///
    /// Intensity amplification factor at focus compared to source surface.
    /// Accounts for geometric focusing and diffraction effects.
    /// For ideal focusing: gain ≈ (aperture_area) / (spot_size²)
    fn focal_gain(&self) -> Option<f64> {
        None
    }

    /// Get comprehensive focal properties
    ///
    /// Convenience method that collects all focal properties into a single struct.
    /// Returns `None` if the source is not focused.
    fn get_focal_properties(&self) -> Option<FocalProperties> {
        // Only return properties if source has a focal point
        let focal_point = self.focal_point()?;
        let focal_depth = self.focal_depth()?;
        let spot_size = self.spot_size()?;

        Some(FocalProperties {
            focal_point,
            focal_depth,
            spot_size,
            f_number: self.f_number(),
            rayleigh_range: self.rayleigh_range(),
            numerical_aperture: self.numerical_aperture(),
            focal_gain: self.focal_gain(),
        })
    }
}
