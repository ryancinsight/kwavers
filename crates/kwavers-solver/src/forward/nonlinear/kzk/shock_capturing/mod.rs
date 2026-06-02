//! Shock Formation Detection and Capturing for KZK Equation
//!
//! Implements shock wave detection and numerical capturing for discontinuous
//! waveforms that develop during nonlinear acoustic propagation.
//!
//! ## Shock Capturing Strategies
//!
//! **Artificial Viscosity**: `Q_av = c₀μ|∇p|/ρ₀ · ∇²p`
//!
//! **Shock Detection via Harmonic Tracking** (Rankine-Hugoniot / Blackstock 1966):
//! monitor pressure steepness and energy at harmonic components.
//!
//! ## References
//! - Blackstock (1966) J. Acoust. Soc. Am. 39(6)
//! - IEEE Std 519-2014 §3.1 (THD via DFT)

use kwavers_core::constants::SOUND_SPEED_TISSUE;

mod capture;
mod detection;
#[cfg(test)]
mod tests;

/// Configuration for shock capturing
#[derive(Debug, Clone, Copy)]
pub struct ShockCapturingConfig {
    /// Enable shock detection
    pub enable_detection: bool,
    /// Enable shock capturing (artificial viscosity)
    pub enable_capturing: bool,
    /// Pressure gradient threshold for shock detection (Pa/m)
    pub gradient_threshold: f64,
    /// Artificial viscosity coefficient (0.0 - 1.0)
    pub viscosity_coefficient: f64,
    /// Harmonic energy threshold for shock detection (ratio to fundamental)
    pub harmonic_threshold: f64,
    /// Number of harmonics to track
    pub num_harmonics: usize,
    /// Window size for gradient calculation (samples)
    pub gradient_window: usize,
    /// Shock velocity estimate (m/s)
    pub shock_velocity_estimate: f64,
}

impl Default for ShockCapturingConfig {
    fn default() -> Self {
        Self {
            enable_detection: true,
            enable_capturing: true,
            gradient_threshold: 1e4,
            viscosity_coefficient: 0.1,
            harmonic_threshold: 0.1,
            num_harmonics: 3,
            gradient_window: 3,
            shock_velocity_estimate: SOUND_SPEED_TISSUE,
        }
    }
}

/// Results from shock detection analysis
#[derive(Debug, Clone)]
pub struct ShockDetectionResult {
    pub shock_detected: bool,
    pub shock_location: Option<usize>,
    pub max_gradient: f64,
    pub steepness_parameter: f64,
    pub harmonic_ratios: Vec<f64>,
    pub thd: f64,
    pub shock_distance: Option<f64>,
    pub shock_strength: Option<f64>,
}

impl Default for ShockDetectionResult {
    fn default() -> Self {
        Self {
            shock_detected: false,
            shock_location: None,
            max_gradient: 0.0,
            steepness_parameter: 0.0,
            harmonic_ratios: Vec::new(),
            thd: 0.0,
            shock_distance: None,
            shock_strength: None,
        }
    }
}

/// Shock formation detector and capturer
#[derive(Debug)]
pub struct ShockCapture {
    config: ShockCapturingConfig,
    history: Vec<ShockDetectionResult>,
}
