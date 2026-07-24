//! Piston source implementation
//!
//! This source models a planar piston transducer that generates
//! acoustic waves from a flat surface.

use aequitas::systems::si::quantities::Length;
use kwavers_grid::Grid;
use kwavers_signal::Signal;
use kwavers_source::{Source, SourceField};
use leto::Array3;
use std::f64::consts::PI;
use std::fmt::Debug;
use std::sync::Arc;

use crate::transducers::physics::CartesianPosition;

/// Piston source configuration
#[derive(Debug, Clone)]
pub struct PistonConfig {
    /// Center position of the piston in SI metres.
    pub center: CartesianPosition,
    /// Diameter of the piston.
    pub diameter: Length,
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
            center: CartesianPosition::from_base([0.0, 0.0, 0.0])
                .expect("invariant: default piston center is finite"),
            diameter: Length::from_base(10.0e-3), // 10mm diameter
            normal: (0.0, 0.0, 1.0),              // Default: z-direction
            source_type: SourceField::Pressure,
            apodization: PistonApodization::Uniform,
        }
    }
}

/// Apodization types for piston sources
#[derive(Debug, Clone, Copy, Default)]
pub enum PistonApodization {
    /// Uniform illumination across the piston
    #[default]
    Uniform,
    /// Gaussian apodization
    Gaussian { sigma: Length },
    /// Cosine apodization (edge tapering)
    Cosine,
    /// Custom apodization function
    Custom {
        function: fn(f64, f64, f64, f64, f64, f64) -> f64,
    },
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
        let radius = config.diameter.into_base() / 2.0;
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
    #[must_use]
    pub fn radius(&self) -> Length {
        Length::from_base(self.radius)
    }

    /// Get the piston diameter
    #[must_use]
    pub fn diameter(&self) -> Length {
        self.config.diameter
    }

    /// Get the piston center position
    #[must_use]
    pub fn center(&self) -> CartesianPosition {
        self.config.center
    }

    /// Calculate apodization weight at a given position
    fn apodization_weight(&self, x: f64, y: f64, z: f64) -> f64 {
        // Calculate distance from center in the piston plane
        let [center_x, center_y, center_z] = self.config.center.into_base();
        let dx = x - center_x;
        let dy = y - center_y;
        let dz = z - center_z;

        // Project onto the piston plane (perpendicular to normal)
        let radial_distance = match self.config.normal {
            (nx, _ny, _nz) if nx.abs() > 0.5 => {
                // Piston is mainly in y-z plane
                dy.hypot(dz)
            }
            (_nx, ny, _nz) if ny.abs() > 0.5 => {
                // Piston is mainly in x-z plane
                dx.hypot(dz)
            }
            _ => {
                // Piston is mainly in x-y plane (default)
                dx.hypot(dy)
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
                let sigma = sigma.into_base();
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
            PistonApodization::Custom { function } => {
                function(x, y, z, center_x, center_y, center_z)
            }
        }
    }
}

impl Source for PistonSource {
    fn create_mask(&self, grid: &Grid) -> Array3<f64> {
        let mut mask = Array3::zeros([grid.nx, grid.ny, grid.nz]);
        self.create_mask_into(grid, &mut mask);
        mask
    }

    fn create_mask_into(&self, grid: &Grid, mask: &mut Array3<f64>) {
        debug_assert_eq!(mask.shape(), [grid.nx, grid.ny, grid.nz]);

        for ([i, j, k], val) in mask.indexed_iter_mut().expect("valid") {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k as f64 * grid.dz;

            *val = self.apodization_weight(x, y, z);
        }
    }

    fn add_mask_into(&self, grid: &Grid, mask: &mut Array3<f64>) {
        debug_assert_eq!(mask.shape(), [grid.nx, grid.ny, grid.nz]);

        for ([i, j, k], val) in mask.indexed_iter_mut().expect("valid") {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k as f64 * grid.dz;

            *val += self.apodization_weight(x, y, z);
        }
    }

    fn amplitude(&self, t: f64) -> f64 {
        self.signal.amplitude(t)
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        // Return the center position
        let [x, y, z] = self.config.center.into_base();
        vec![(x, y, z)]
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
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn center(mut self, center: CartesianPosition) -> Self {
        self.config.center = center;
        self
    }

    #[must_use]
    pub fn diameter(mut self, diameter: Length) -> Self {
        self.config.diameter = diameter;
        self
    }

    #[must_use]
    pub fn normal(mut self, normal: (f64, f64, f64)) -> Self {
        self.config.normal = normal;
        self
    }

    #[must_use]
    pub fn source_type(mut self, source_type: SourceField) -> Self {
        self.config.source_type = source_type;
        self
    }

    #[must_use]
    pub fn apodization(mut self, apodization: PistonApodization) -> Self {
        self.config.apodization = apodization;
        self
    }

    pub fn build(self, signal: Arc<dyn Signal>) -> PistonSource {
        PistonSource::new(self.config, signal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_signal::NullSignal;

    #[test]
    fn typed_geometry_preserves_si_values() {
        let center =
            CartesianPosition::from_base([1.0e-3, -2.0e-3, 3.0e-3]).expect("finite test position");
        let source = PistonBuilder::new()
            .center(center)
            .diameter(Length::from_base(4.0e-3))
            .apodization(PistonApodization::Gaussian {
                sigma: Length::from_base(1.0e-3),
            })
            .build(Arc::new(NullSignal));

        assert_eq!(source.center().into_base(), [1.0e-3, -2.0e-3, 3.0e-3]);
        assert_eq!(source.diameter().into_base(), 4.0e-3);
        assert_eq!(source.radius().into_base(), 2.0e-3);
        assert_eq!(source.apodization_weight(1.0e-3, -2.0e-3, 3.0e-3), 1.0);
    }
}
