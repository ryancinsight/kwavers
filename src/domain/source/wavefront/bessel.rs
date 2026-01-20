//! Bessel beam source implementation
//!
//! This source generates non-diffracting Bessel beams that maintain
//! their shape over long distances, useful for applications requiring
//! extended depth of field.

use crate::domain::grid::Grid;
use crate::domain::signal::Signal;
use crate::domain::source::{Source, SourceField};
use ndarray::Array3;
use std::f64::consts::PI;
use std::fmt::Debug;
use std::sync::Arc;

/// Bessel beam configuration
#[derive(Debug, Clone)]
pub struct BesselConfig {
    /// Center position [x, y, z] in meters
    pub center: (f64, f64, f64),
    /// Propagation direction (unit vector)
    pub direction: (f64, f64, f64),
    /// Wavelength in meters
    pub wavelength: f64,
    /// Radial wave number (k_r) in rad/m
    pub radial_wavenumber: f64,
    /// Axial wave number (k_z) in rad/m
    pub axial_wavenumber: f64,
    /// Order of Bessel function (n)
    pub order: usize,
    /// Source field type
    pub source_type: SourceField,
    /// Phase offset in radians
    pub phase: f64,
}

impl Default for BesselConfig {
    fn default() -> Self {
        let wavelength = 1.5e-3_f64; // 1mm wavelength
        let radial_wavenumber = 1000.0_f64; // k_r = 1000 rad/m
        let axial_wavenumber = ((2.0 * PI / wavelength).powi(2) - radial_wavenumber.powi(2)).sqrt();

        Self {
            center: (0.0, 0.0, 0.0),
            direction: (0.0, 0.0, 1.0), // Default: z-direction
            wavelength,
            radial_wavenumber,
            axial_wavenumber,
            order: 0, // Zeroth-order Bessel beam (most common)
            source_type: SourceField::Pressure,
            phase: 0.0,
        }
    }
}

/// Bessel beam source implementation
#[derive(Debug)]
pub struct BesselSource {
    config: BesselConfig,
    signal: Arc<dyn Signal>,
    #[allow(dead_code)]
    wave_number: f64,
}

impl BesselSource {
    /// Create a new Bessel beam source
    pub fn new(config: BesselConfig, signal: Arc<dyn Signal>) -> Self {
        let wave_number = 2.0 * PI / config.wavelength;

        Self {
            config,
            signal,
            wave_number,
        }
    }

    /// Create a Bessel source with default configuration
    pub fn new_default(signal: Arc<dyn Signal>) -> Self {
        Self::new(BesselConfig::default(), signal)
    }

    /// Get the radial wave number (k_r)
    pub fn radial_wavenumber(&self) -> f64 {
        self.config.radial_wavenumber
    }

    /// Get the axial wave number (k_z)
    pub fn axial_wavenumber(&self) -> f64 {
        self.config.axial_wavenumber
    }

    /// Get the Bessel function order
    pub fn order(&self) -> usize {
        self.config.order
    }

    /// Calculate Bessel function of order n at position r
    fn bessel_j(&self, n: usize, r: f64) -> f64 {
        // Implement Bessel function using series expansion
        // This is a simplified implementation - for production use, consider using
        // a more accurate numerical library

        if r == 0.0 {
            if n == 0 {
                return 1.0; // J0(0) = 1
            } else {
                return 0.0; // Jn(0) = 0 for n > 0
            }
        }

        let x = self.config.radial_wavenumber * r;
        let mut result = 0.0;

        // Series expansion for Bessel function
        // Jn(x) = sum_{k=0}^∞ (-1)^k / (k! (n+k)!) * (x/2)^(2k+n)

        for k in 0..20 {
            // Limit to 20 terms for performance
            let term1 = (-1.0_f64).powi(k as i32);
            let term2 = 1.0 / (Self::factorial(k) * Self::factorial(n + k));
            let term3 = (x / 2.0).powi((2 * k + n) as i32);
            result += term1 * term2 * term3;
        }

        result
    }

    /// Calculate Bessel beam amplitude at position (x, y, z)
    fn bessel_amplitude(&self, x: f64, y: f64, z: f64) -> f64 {
        // Calculate distance from center
        let dx = x - self.config.center.0;
        let dy = y - self.config.center.1;
        let dz = z - self.config.center.2;

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

        // Bessel function value
        let bessel_value = self.bessel_j(self.config.order, radial_distance);

        // Phase term
        let phase_term = self.config.axial_wavenumber * z_dist + self.config.phase;

        // Total amplitude
        bessel_value * phase_term.cos()
    }

    /// Helper function to calculate factorial
    fn factorial(n: usize) -> f64 {
        if n == 0 {
            1.0
        } else {
            (1..=n).product::<usize>() as f64
        }
    }
}

impl Source for BesselSource {
    fn create_mask(&self, grid: &Grid) -> Array3<f64> {
        let mut mask = Array3::zeros((grid.nx, grid.ny, grid.nz));

        for ((i, j, k), val) in mask.indexed_iter_mut() {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k as f64 * grid.dz;

            *val = self.bessel_amplitude(x, y, z);
        }

        mask
    }

    fn amplitude(&self, t: f64) -> f64 {
        self.signal.amplitude(t)
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        // Return the center position
        vec![self.config.center]
    }

    fn signal(&self) -> &dyn Signal {
        self.signal.as_ref()
    }

    fn source_type(&self) -> SourceField {
        self.config.source_type
    }

    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, _grid: &Grid) -> f64 {
        let spatial_amplitude = self.bessel_amplitude(x, y, z);
        let temporal_amplitude = self.signal.amplitude(t);
        spatial_amplitude * temporal_amplitude
    }
}

/// Builder pattern for Bessel beam source
#[derive(Debug, Default)]
pub struct BesselBuilder {
    config: BesselConfig,
}

impl BesselBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn center(mut self, center: (f64, f64, f64)) -> Self {
        self.config.center = center;
        self
    }

    pub fn direction(mut self, direction: (f64, f64, f64)) -> Self {
        self.config.direction = direction;
        self
    }

    pub fn wavelength(mut self, wavelength: f64) -> Self {
        self.config.wavelength = wavelength;
        // Recalculate axial wave number when wavelength changes
        self.config.axial_wavenumber =
            ((2.0 * PI / wavelength).powi(2) - self.config.radial_wavenumber.powi(2)).sqrt();
        self
    }

    pub fn radial_wavenumber(mut self, radial_wavenumber: f64) -> Self {
        self.config.radial_wavenumber = radial_wavenumber;
        // Recalculate axial wave number when radial wave number changes
        self.config.axial_wavenumber =
            ((2.0 * PI / self.config.wavelength).powi(2) - radial_wavenumber.powi(2)).sqrt();
        self
    }

    pub fn order(mut self, order: usize) -> Self {
        self.config.order = order;
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

    pub fn build(self, signal: Arc<dyn Signal>) -> BesselSource {
        BesselSource::new(self.config, signal)
    }
}
