//! Piston source implementation
//!
//! This source models a planar piston transducer that generates
//! acoustic waves from a flat surface.

use crate::grid::Grid;
use crate::signal::Signal;
use crate::source::{Source, SourceField};
use ndarray::Array3;
use std::f64::consts::PI;
use std::fmt::Debug;
use std::sync::Arc;

/// Piston source configuration
#[derive(Debug, Clone)]
pub struct PistonConfig {
    /// Center position of the piston [x, y, z] in meters
    pub center: (f64, f64, f64),
    /// Diameter of the piston in meters
    pub diameter: f64,
    /// Normal direction (unit vector pointing outward)
    pub normal: (f64, f64, f64),
    /// Source field type
    pub source_type: SourceField,
    /// Apodization function (uniform by default)
    pub apodization: PistonApodization,
}

impl Default for PistonConfig {
    fn default() -> Self {
        Self {
            center: (0.0, 0.0, 0.0),
            diameter: 10.0e-3,       // 10mm diameter
            normal: (0.0, 0.0, 1.0), // Default: z-direction
            source_type: SourceField::Pressure,
            apodization: PistonApodization::Uniform,
        }
    }
}

/// Apodization types for piston sources
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PistonApodization {
    /// Uniform illumination across the piston
    Uniform,
    /// Gaussian apodization
    Gaussian { sigma: f64 },
    /// Cosine apodization (edge tapering)
    Cosine,
    /// Custom apodization function
    Custom {
        function: fn(f64, f64, f64, f64, f64, f64) -> f64,
    },
}

impl Default for PistonApodization {
    fn default() -> Self {
        Self::Uniform
    }
}

/// Piston source implementation
#[derive(Debug)]
pub struct PistonSource {
    config: PistonConfig,
    signal: Arc<dyn Signal>,
    radius: f64,
}

impl PistonSource {
    /// Create a new piston source
    pub fn new(config: PistonConfig, signal: Arc<dyn Signal>) -> Self {
        let radius = config.diameter / 2.0;
        Self {
            config,
            signal,
            radius,
        }
    }

    /// Create a piston source with default configuration
    pub fn new_default(signal: Arc<dyn Signal>) -> Self {
        Self::new(PistonConfig::default(), signal)
    }

    /// Get the piston radius
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Get the piston diameter
    pub fn diameter(&self) -> f64 {
        self.config.diameter
    }

    /// Get the piston center position
    pub fn center(&self) -> (f64, f64, f64) {
        self.config.center
    }

    /// Calculate apodization weight at a given position
    fn apodization_weight(&self, x: f64, y: f64, z: f64) -> f64 {
        // Calculate distance from center in the piston plane
        let dx = x - self.config.center.0;
        let dy = y - self.config.center.1;
        let dz = z - self.config.center.2;

        // Project onto the piston plane (perpendicular to normal)
        let radial_distance = match self.config.normal {
            (nx, ny, nz) if nx.abs() > 0.5 => {
                // Piston is mainly in y-z plane
                (dy.powi(2) + dz.powi(2)).sqrt()
            }
            (nx, ny, nz) if ny.abs() > 0.5 => {
                // Piston is mainly in x-z plane
                (dx.powi(2) + dz.powi(2)).sqrt()
            }
            _ => {
                // Piston is mainly in x-y plane (default)
                (dx.powi(2) + dy.powi(2)).sqrt()
            }
        };

        match self.config.apodization {
            PistonApodization::Uniform => {
                if radial_distance <= self.radius {
                    1.0
                } else {
                    0.0
                }
            }
            PistonApodization::Gaussian { sigma } => {
                if radial_distance <= self.radius {
                    (-radial_distance.powi(2) / (2.0 * sigma.powi(2))).exp()
                } else {
                    0.0
                }
            }
            PistonApodization::Cosine => {
                if radial_distance <= self.radius {
                    (PI * radial_distance / self.radius).cos()
                } else {
                    0.0
                }
            }
            PistonApodization::Custom { function } => function(
                x,
                y,
                z,
                self.config.center.0,
                self.config.center.1,
                self.config.center.2,
            ),
        }
    }
}

impl Source for PistonSource {
    fn create_mask(&self, grid: &Grid) -> Array3<f64> {
        let mut mask = Array3::zeros((grid.nx, grid.ny, grid.nz));

        for ((i, j, k), val) in mask.indexed_iter_mut() {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k as f64 * grid.dz;

            *val = self.apodization_weight(x, y, z);
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
        let weight = self.apodization_weight(x, y, z);
        weight * self.signal.amplitude(t)
    }
}

/// Builder pattern for piston source
#[derive(Debug, Default)]
pub struct PistonBuilder {
    config: PistonConfig,
}

impl PistonBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn center(mut self, center: (f64, f64, f64)) -> Self {
        self.config.center = center;
        self
    }

    pub fn diameter(mut self, diameter: f64) -> Self {
        self.config.diameter = diameter;
        self
    }

    pub fn normal(mut self, normal: (f64, f64, f64)) -> Self {
        self.config.normal = normal;
        self
    }

    pub fn source_type(mut self, source_type: SourceField) -> Self {
        self.config.source_type = source_type;
        self
    }

    pub fn apodization(mut self, apodization: PistonApodization) -> Self {
        self.config.apodization = apodization;
        self
    }

    pub fn build(self, signal: Arc<dyn Signal>) -> PistonSource {
        PistonSource::new(self.config, signal)
    }
}
