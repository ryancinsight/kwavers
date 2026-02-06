//! Fiber optic source implementations
//!
//! This module provides fiber optic light sources for optical simulations.

use crate::domain::source::optical::OpticalSource;

/// Fiber optic configuration
#[derive(Debug, Clone)]
pub struct FiberConfig {
    /// Core diameter (m)
    pub core_diameter: f64,
    /// Numerical aperture
    pub numerical_aperture: f64,
    /// Wavelength (m)
    pub wavelength: f64,
    /// Output power (W)
    pub output_power: f64,
}

impl Default for FiberConfig {
    fn default() -> Self {
        Self {
            core_diameter: 50e-6, // 50 μm core
            numerical_aperture: 0.22,
            wavelength: 850e-9, // 850 nm
            output_power: 0.1,  // 100 mW
        }
    }
}

/// Fiber optic source
#[derive(Debug)]
pub struct FiberSource {
    config: FiberConfig,
    position: (f64, f64, f64),
}

impl FiberSource {
    /// Create a new fiber optic source
    pub fn new(config: FiberConfig, position: (f64, f64, f64)) -> Self {
        Self { config, position }
    }
}

impl OpticalSource for FiberSource {
    fn optical_power(&self, _t: f64) -> f64 {
        self.config.output_power
    }

    fn wavelength(&self) -> f64 {
        self.config.wavelength
    }

    fn beam_profile(&self, x: f64, y: f64, z: f64) -> f64 {
        // Simple Gaussian beam profile for fiber output
        let dx = x - self.position.0;
        let dy = y - self.position.1;
        let dz = z - self.position.2;
        let r = (dx * dx + dy * dy + dz * dz).sqrt();

        let radius = self.config.core_diameter / 2.0;
        (-r * r / (radius * radius)).exp()
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        vec![self.position]
    }
}
