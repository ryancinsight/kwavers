//! Gaussian beam source implementation
//!
//! This source generates focused Gaussian beams commonly used in
//! medical imaging and optical applications.

use crate::grid::Grid;
use crate::signal::Signal;
use crate::source::{Source, SourceField};
use ndarray::Array3;
use std::f64::consts::PI;
use std::fmt::Debug;
use std::sync::Arc;

/// Gaussian beam configuration
#[derive(Debug, Clone)]
pub struct GaussianConfig {
    /// Focal point position [x, y, z] in meters
    pub focal_point: (f64, f64, f64),
    /// Beam waist radius at focus (w0) in meters
    pub waist_radius: f64,
    /// Wavelength in meters
    pub wavelength: f64,
    /// Propagation direction (unit vector)
    pub direction: (f64, f64, f64),
    /// Source field type
    pub source_type: SourceField,
    /// Phase offset in radians
    pub phase: f64,
}

impl Default for GaussianConfig {
    fn default() -> Self {
        Self {
            focal_point: (0.0, 0.0, 0.0),
            waist_radius: 1.0e-3,       // 1mm waist radius
            wavelength: 1.5e-3,         // 1mm wavelength (1.5MHz in water)
            direction: (0.0, 0.0, 1.0), // Default: z-direction
            source_type: SourceField::Pressure,
            phase: 0.0,
        }
    }
}

/// Gaussian beam source implementation
#[derive(Debug)]
pub struct GaussianSource {
    config: GaussianConfig,
    signal: Arc<dyn Signal>,
    wave_number: f64,
    rayleigh_range: f64,
}

impl GaussianSource {
    /// Create a new Gaussian beam source
    pub fn new(config: GaussianConfig, signal: Arc<dyn Signal>) -> Self {
        let wave_number = 2.0 * PI / config.wavelength;
        let rayleigh_range = PI * config.waist_radius.powi(2) / config.wavelength;

        Self {
            config,
            signal,
            wave_number,
            rayleigh_range,
        }
    }

    /// Create a Gaussian source with default configuration
    pub fn new_default(signal: Arc<dyn Signal>) -> Self {
        Self::new(GaussianConfig::default(), signal)
    }

    /// Get the beam waist radius (w0)
    pub fn waist_radius(&self) -> f64 {
        self.config.waist_radius
    }

    /// Get the Rayleigh range (depth of focus)
    pub fn rayleigh_range(&self) -> f64 {
        self.rayleigh_range
    }

    /// Get the focal point position
    pub fn focal_point(&self) -> (f64, f64, f64) {
        self.config.focal_point
    }

    /// Calculate beam radius at distance z from focus
    fn beam_radius_at(&self, z: f64) -> f64 {
        self.config.waist_radius * (1.0 + (z / self.rayleigh_range).powi(2)).sqrt()
    }

    /// Calculate Gaussian amplitude at position (x, y, z)
    fn gaussian_amplitude(&self, x: f64, y: f64, z: f64) -> f64 {
        // Calculate distance from focal point
        let dx = x - self.config.focal_point.0;
        let dy = y - self.config.focal_point.1;
        let dz = z - self.config.focal_point.2;

        // Project onto the propagation plane (perpendicular to direction)
        let radial_distance = match self.config.direction {
            (nx, ny, nz) if nz.abs() > 0.5 => {
                // Mainly z-propagation: use x-y plane
                (dx.powi(2) + dy.powi(2)).sqrt()
            }
            (nx, ny, nz) if ny.abs() > 0.5 => {
                // Mainly y-propagation: use x-z plane
                (dx.powi(2) + dz.powi(2)).sqrt()
            }
            _ => {
                // Mainly x-propagation: use y-z plane
                (dy.powi(2) + dz.powi(2)).sqrt()
            }
        };

        // Distance along propagation direction
        let z_dist = match self.config.direction {
            (nx, ny, nz) if nz.abs() > 0.5 => dz,
            (nx, ny, nz) if ny.abs() > 0.5 => dy,
            _ => dx,
        };

        // Beam radius at this z position
        let w_z = self.beam_radius_at(z_dist);

        // Gaussian amplitude profile
        let amplitude = (-radial_distance.powi(2) / w_z.powi(2)).exp();

        // Gouy phase shift
        let gouy_phase = (z_dist / self.rayleigh_range).atan();

        // Total phase including propagation and Gouy phase
        let total_phase = self.wave_number * z_dist + gouy_phase + self.config.phase;

        amplitude * total_phase.cos()
    }
}

impl Source for GaussianSource {
    fn create_mask(&self, grid: &Grid) -> Array3<f64> {
        let mut mask = Array3::zeros((grid.nx, grid.ny, grid.nz));

        for ((i, j, k), val) in mask.indexed_iter_mut() {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k as f64 * grid.dz;

            *val = self.gaussian_amplitude(x, y, z);
        }

        mask
    }

    fn amplitude(&self, t: f64) -> f64 {
        self.signal.amplitude(t)
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        // Return the focal point
        vec![self.config.focal_point]
    }

    fn signal(&self) -> &dyn Signal {
        self.signal.as_ref()
    }

    fn source_type(&self) -> SourceField {
        self.config.source_type
    }

    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, _grid: &Grid) -> f64 {
        let spatial_amplitude = self.gaussian_amplitude(x, y, z);
        let temporal_amplitude = self.signal.amplitude(t);
        spatial_amplitude * temporal_amplitude
    }
}

/// Builder pattern for Gaussian beam source
#[derive(Debug, Default)]
pub struct GaussianBuilder {
    config: GaussianConfig,
}

impl GaussianBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn focal_point(mut self, focal_point: (f64, f64, f64)) -> Self {
        self.config.focal_point = focal_point;
        self
    }

    pub fn waist_radius(mut self, waist_radius: f64) -> Self {
        self.config.waist_radius = waist_radius;
        self
    }

    pub fn wavelength(mut self, wavelength: f64) -> Self {
        self.config.wavelength = wavelength;
        self
    }

    pub fn direction(mut self, direction: (f64, f64, f64)) -> Self {
        self.config.direction = direction;
        self
    }

    pub fn source_type(mut self, source_type: SourceField) -> Self {
        self.config.source_type = source_type;
        self
    }

    pub fn phase(mut self, phase: f64) -> Self {
        self.config.phase = phase;
        self
    }

    pub fn build(self, signal: Arc<dyn Signal>) -> GaussianSource {
        GaussianSource::new(self.config, signal)
    }
}
