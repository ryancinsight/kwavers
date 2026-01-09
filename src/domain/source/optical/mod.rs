//! Optical source types
//!
//! This module contains optical source implementations for light propagation
//! simulations, including laser sources, LED arrays, and fiber optics.

pub mod fiber;
pub mod laser;
pub mod led;

pub use fiber::{FiberConfig, FiberSource};
pub use laser::{GaussianLaser, LaserConfig, LaserSource};
pub use led::{LEDConfig, LEDSource};

/// Optical source trait
pub trait OpticalSource {
    /// Get the optical power at a given time (W)
    fn optical_power(&self, t: f64) -> f64;

    /// Get the wavelength (m)
    fn wavelength(&self) -> f64;

    /// Get the beam profile
    fn beam_profile(&self, x: f64, y: f64, z: f64) -> f64;

    /// Get source positions
    fn positions(&self) -> Vec<(f64, f64, f64)>;
}
