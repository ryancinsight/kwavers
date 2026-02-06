//! Spherical wave source implementation
//!
//! This source generates spherical waves that can be diverging (from a point)
//! or converging (toward a point), useful for modeling point sources and
//! focused wave fields.

use crate::domain::grid::Grid;
use crate::domain::signal::Signal;
use crate::domain::source::{Source, SourceField};
use ndarray::Array3;
use std::f64::consts::PI;
use std::fmt::Debug;
use std::sync::Arc;

/// Spherical wave configuration
#[derive(Debug, Clone)]
pub struct SphericalConfig {
    /// Center position [x, y, z] in meters
    pub center: (f64, f64, f64),
    /// Wavelength in meters
    pub wavelength: f64,
    /// Wave type (diverging or converging)
    pub wave_type: SphericalWaveType,
    /// Source field type
    pub source_type: SourceField,
    /// Phase offset in radians
    pub phase: f64,
    /// Attenuation coefficient (1/m)
    pub attenuation: f64,
}

impl Default for SphericalConfig {
    fn default() -> Self {
        Self {
            center: (0.0, 0.0, 0.0),
            wavelength: 1.5e-3, // 1mm wavelength (1.5MHz in water)
            wave_type: SphericalWaveType::Diverging,
            source_type: SourceField::Pressure,
            phase: 0.0,
            attenuation: 0.0, // No attenuation by default
        }
    }
}

/// Type of spherical wave
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SphericalWaveType {
    /// Diverging wave (outward from center)
    #[default]
    Diverging,
    /// Converging wave (inward toward center)
    Converging,
}

/// Spherical wave source implementation
#[derive(Debug)]
pub struct SphericalSource {
    config: SphericalConfig,
    signal: Arc<dyn Signal>,
    wave_number: f64,
}

impl SphericalSource {
    /// Create a new spherical wave source
    pub fn new(config: SphericalConfig, signal: Arc<dyn Signal>) -> Self {
        let wave_number = 2.0 * PI / config.wavelength;

        Self {
            config,
            signal,
            wave_number,
        }
    }

    /// Create a spherical source with default configuration
    pub fn new_default(signal: Arc<dyn Signal>) -> Self {
        Self::new(SphericalConfig::default(), signal)
    }

    /// Get the wave number (k = 2π/λ)
    pub fn wave_number(&self) -> f64 {
        self.wave_number
    }

    /// Get the wave type
    pub fn wave_type(&self) -> SphericalWaveType {
        self.config.wave_type
    }

    /// Calculate spherical wave amplitude at position (x, y, z)
    fn spherical_amplitude(&self, x: f64, y: f64, z: f64) -> f64 {
        // Calculate distance from center
        let dx = x - self.config.center.0;
        let dy = y - self.config.center.1;
        let dz = z - self.config.center.2;
        let distance = (dx.powi(2) + dy.powi(2) + dz.powi(2)).sqrt();

        // Handle singularity at center
        if distance == 0.0 {
            return 1.0; // Maximum amplitude at center
        }

        // Amplitude decay (1/r for spherical waves)
        let amplitude = 1.0 / distance;

        // Phase term
        let phase_term = match self.config.wave_type {
            SphericalWaveType::Diverging => self.wave_number * distance,
            SphericalWaveType::Converging => -self.wave_number * distance,
        } + self.config.phase;

        // Attenuation
        let attenuation_factor = (-self.config.attenuation * distance).exp();

        // Total amplitude
        amplitude * phase_term.cos() * attenuation_factor
    }
}

impl Source for SphericalSource {
    fn create_mask(&self, grid: &Grid) -> Array3<f64> {
        let mut mask = Array3::zeros((grid.nx, grid.ny, grid.nz));

        for ((i, j, k), val) in mask.indexed_iter_mut() {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k as f64 * grid.dz;

            *val = self.spherical_amplitude(x, y, z);
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
        let spatial_amplitude = self.spherical_amplitude(x, y, z);
        let temporal_amplitude = self.signal.amplitude(t);
        spatial_amplitude * temporal_amplitude
    }
}

/// Builder pattern for spherical wave source
#[derive(Debug, Default)]
pub struct SphericalBuilder {
    config: SphericalConfig,
}

impl SphericalBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn center(mut self, center: (f64, f64, f64)) -> Self {
        self.config.center = center;
        self
    }

    pub fn wavelength(mut self, wavelength: f64) -> Self {
        self.config.wavelength = wavelength;
        self
    }

    pub fn wave_type(mut self, wave_type: SphericalWaveType) -> Self {
        self.config.wave_type = wave_type;
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

    pub fn attenuation(mut self, attenuation: f64) -> Self {
        self.config.attenuation = attenuation;
        self
    }

    pub fn build(self, signal: Arc<dyn Signal>) -> SphericalSource {
        SphericalSource::new(self.config, signal)
    }
}
