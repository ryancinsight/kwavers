//! Laser source implementations
//!
//! This module provides Gaussian laser beam sources for optical simulations.

use crate::domain::signal::{Signal, TimeVaryingSignal};

/// Laser configuration
#[derive(Debug, Clone)]
pub struct LaserConfig {
    /// Wavelength (m)
    pub wavelength: f64,
    /// Beam waist radius (m)
    pub beam_waist: f64,
    /// Peak power (W)
    pub peak_power: f64,
    /// Pulse duration (s)
    pub pulse_duration: f64,
    /// Repetition rate (Hz)
    pub repetition_rate: f64,
}

impl Default for LaserConfig {
    fn default() -> Self {
        Self {
            wavelength: 800e-9,   // 800 nm (near-infrared)
            beam_waist: 1e-3,     // 1 mm beam waist
            peak_power: 1.0,      // 1 W peak power
            pulse_duration: 1e-9, // 1 ns pulse
            repetition_rate: 1e6, // 1 MHz repetition rate
        }
    }
}

/// Gaussian laser source
#[derive(Debug)]
pub struct GaussianLaser {
    config: LaserConfig,
    position: (f64, f64, f64),
    direction: (f64, f64, f64),
    signal: TimeVaryingSignal,
}

impl GaussianLaser {
    /// Create a new Gaussian laser source
    pub fn new(config: LaserConfig, position: (f64, f64, f64), direction: (f64, f64, f64)) -> Self {
        // Create pulse signal
        let pulse_signal = vec![
            1.0; // Simple square pulse for now
            (config.pulse_duration / 1e-12) as usize // Convert to timesteps
        ];

        let signal = TimeVaryingSignal::new(pulse_signal, 1e-12); // 1 ps timestep

        Self {
            config,
            position,
            direction,
            signal,
        }
    }

    /// Get beam profile at a point
    pub fn beam_profile(&self, x: f64, y: f64, z: f64) -> f64 {
        // Calculate distance from beam axis
        let dx = x - self.position.0;
        let dy = y - self.position.1;
        let dz = z - self.position.2;

        // Project onto plane perpendicular to beam direction
        let dot = dx * self.direction.0 + dy * self.direction.1 + dz * self.direction.2;
        let proj_x = dx - dot * self.direction.0;
        let proj_y = dy - dot * self.direction.1;
        let proj_z = dz - dot * self.direction.2;

        let r = (proj_x * proj_x + proj_y * proj_y + proj_z * proj_z).sqrt();

        // Gaussian beam profile
        let waist_squared = self.config.beam_waist * self.config.beam_waist;
        (-r * r / (2.0 * waist_squared)).exp()
    }
}

impl super::OpticalSource for GaussianLaser {
    fn optical_power(&self, t: f64) -> f64 {
        self.config.peak_power * self.signal.amplitude(t)
    }

    fn wavelength(&self) -> f64 {
        self.config.wavelength
    }

    fn beam_profile(&self, x: f64, y: f64, z: f64) -> f64 {
        self.beam_profile(x, y, z)
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        vec![self.position]
    }
}

/// Laser source implementation
#[derive(Debug)]
pub struct LaserSource {
    #[allow(dead_code)]
    laser: GaussianLaser,
}

impl LaserSource {
    pub fn new(config: LaserConfig, position: (f64, f64, f64), direction: (f64, f64, f64)) -> Self {
        Self {
            laser: GaussianLaser::new(config, position, direction),
        }
    }
}
