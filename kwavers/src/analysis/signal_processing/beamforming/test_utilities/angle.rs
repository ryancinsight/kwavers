//! Angle conversion utilities and constants.

use std::f64::consts::PI;

/// Convert degrees to radians.
#[inline]
pub fn deg_to_rad(deg: f64) -> f64 {
    deg * PI / 180.0
}

/// Convert radians to degrees.
#[inline]
pub fn rad_to_deg(rad: f64) -> f64 {
    rad * 180.0 / PI
}

/// Broadside angle (0 radians).
pub const BROADSIDE: f64 = 0.0;

/// Positive endfire angle (π/2 radians, 90°).
pub const ENDFIRE_POS: f64 = PI / 2.0;

/// Negative endfire angle (-π/2 radians, -90°).
pub const ENDFIRE_NEG: f64 = -PI / 2.0;
