//! Core phase shifting utilities and constants
//!
//! This module provides fundamental phase calculation functions and constants
//! used throughout the phase shifting subsystem. Implements SSOT principle
//! for phase-related computations.

use crate::constants::physics::SOUND_SPEED_WATER;
use std::f64::consts::{PI, TAU};

/// Maximum number of focal points supported
pub const MAX_FOCAL_POINTS: usize = 16;

/// Maximum steering angle in radians
pub const MAX_STEERING_ANGLE: f64 = PI / 3.0; // 60 degrees

/// Minimum focal distance in meters
pub const MIN_FOCAL_DISTANCE: f64 = 0.001; // 1mm

/// Default speed of sound for phase calculations
pub const SPEED_OF_SOUND: f64 = SOUND_SPEED_WATER;

/// Phase shifting strategy enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShiftingStrategy {
    /// Linear phase gradient for beam steering
    Linear,
    /// Focused beam with spherical phase front
    Focused,
    /// Multiple focal points
    MultiFocus,
    /// Arbitrary phase pattern
    Custom,
}

/// Calculate wavelength from frequency and sound speed
///
/// # Arguments
/// * `frequency` - Frequency in Hz
/// * `sound_speed` - Speed of sound in m/s
///
/// # Returns
/// Wavelength in meters
#[inline]
#[must_use]
pub fn calculate_wavelength(frequency: f64, sound_speed: f64) -> f64 {
    sound_speed / frequency
}

/// Wrap phase to [-π, π] range
///
/// Uses optimized modulo operation for phase wrapping.
/// Ensures phase continuity for array calculations.
///
/// # Arguments
/// * `phase` - Phase in radians
///
/// # Returns
/// Wrapped phase in [-π, π] range
#[inline]
#[must_use]
pub fn wrap_phase(phase: f64) -> f64 {
    // Wrap phase to [-π, π] range
    let mut p = phase % TAU;

    // Normalize to [-2π, 2π]
    if p > PI {
        p -= TAU;
    } else if p < -PI {
        p += TAU;
    }

    p
}

/// Normalize phase to [0, 2π] range
///
/// # Arguments
/// * `phase` - Phase in radians
///
/// # Returns
/// Normalized phase in [0, 2π] range
#[inline]
#[must_use]
pub fn normalize_phase(phase: f64) -> f64 {
    let normalized = phase % TAU;
    if normalized < 0.0 {
        normalized + TAU
    } else {
        normalized
    }
}

/// Quantize phase to discrete levels
///
/// Used for digital phase shifter implementations with
/// limited phase resolution.
///
/// # Arguments
/// * `phase` - Phase in radians
/// * `levels` - Number of quantization levels
///
/// # Returns
/// Quantized phase value
#[inline]
#[must_use]
pub fn quantize_phase(phase: f64, levels: u32) -> f64 {
    let normalized = normalize_phase(phase);
    let step = TAU / f64::from(levels);
    let quantized_level = (normalized / step).round() as u32;
    f64::from(quantized_level % levels) * step
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_calculate_wavelength() {
        let wavelength = calculate_wavelength(1e6, 1500.0);
        assert_relative_eq!(wavelength, 0.0015, epsilon = 1e-6);
    }

    #[test]
    fn test_wrap_phase() {
        assert_relative_eq!(wrap_phase(0.0), 0.0);
        assert_relative_eq!(wrap_phase(PI), PI);
        assert_relative_eq!(wrap_phase(-PI), -PI);
        assert_relative_eq!(wrap_phase(3.0 * PI), PI, epsilon = 1e-10);
        assert_relative_eq!(wrap_phase(-3.0 * PI), -PI, epsilon = 1e-10);
        // Test wrap around cases
        assert_relative_eq!(wrap_phase(2.0 * PI), 0.0, epsilon = 1e-10);
        assert_relative_eq!(wrap_phase(-2.0 * PI), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_normalize_phase() {
        assert_relative_eq!(normalize_phase(0.0), 0.0);
        assert_relative_eq!(normalize_phase(PI), PI);
        assert_relative_eq!(normalize_phase(-PI), PI);
        assert_relative_eq!(normalize_phase(3.0 * PI), PI, epsilon = 1e-10);
    }

    #[test]
    fn test_quantize_phase() {
        // 4-level quantization: 0, π/2, π, 3π/2
        assert_relative_eq!(quantize_phase(0.1, 4), 0.0);
        assert_relative_eq!(quantize_phase(PI / 3.0, 4), PI / 2.0, epsilon = 1e-10);
        assert_relative_eq!(quantize_phase(PI, 4), PI, epsilon = 1e-10);
    }
}
