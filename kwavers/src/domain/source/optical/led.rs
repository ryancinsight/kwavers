//! LED source implementations
//!
//! This module provides LED array light sources for optical simulations.

use crate::domain::source::optical::OpticalSource;

/// LED configuration
#[derive(Debug, Clone)]
pub struct LEDConfig {
    /// Wavelength (m)
    pub wavelength: f64,
    /// Power per LED (W)
    pub power_per_led: f64,
    /// Beam angle (radians)
    pub beam_angle: f64,
}

impl Default for LEDConfig {
    fn default() -> Self {
        Self {
            wavelength: 630e-9,  // 630 nm (red)
            power_per_led: 0.01, // 10 mW per LED
            beam_angle: 0.5,     // ~30 degrees
        }
    }
}

/// LED array source
#[derive(Debug)]
pub struct LEDSource {
    config: LEDConfig,
    positions: Vec<(f64, f64, f64)>,
}

impl LEDSource {
    /// Create a new LED array source
    pub fn new(config: LEDConfig, positions: Vec<(f64, f64, f64)>) -> Self {
        Self { config, positions }
    }
}

impl OpticalSource for LEDSource {
    fn optical_power(&self, _t: f64) -> f64 {
        self.config.power_per_led * self.positions.len() as f64
    }

    fn wavelength(&self) -> f64 {
        self.config.wavelength
    }

    fn beam_profile(&self, x: f64, y: f64, z: f64) -> f64 {
        // Simple Lambertian beam profile
        let mut total = 0.0;
        for pos in &self.positions {
            let dx = x - pos.0;
            let dy = y - pos.1;
            let dz = z - pos.2;
            let r = (dx * dx + dy * dy + dz * dz).sqrt();

            if r > 0.0 {
                let angle = (dz / r).acos(); // Angle from normal
                if angle <= self.config.beam_angle {
                    // Lambertian cosine distribution
                    let intensity = angle.cos();
                    total += intensity / (r * r); // Inverse square law
                }
            }
        }
        total
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        self.positions.clone()
    }
}
