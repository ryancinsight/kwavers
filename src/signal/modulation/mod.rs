//! Signal modulation techniques
//!
//! This module provides various modulation schemes for signal processing.
//! Each modulation type is in its own submodule for better separation of concerns.

pub mod amplitude;
pub mod frequency;
pub mod phase;
pub mod quadrature;
pub mod pulse_width;

// Re-export main types
pub use amplitude::AmplitudeModulation;
pub use frequency::FrequencyModulation;
pub use phase::PhaseModulation;
pub use quadrature::QuadratureAmplitudeModulation;
pub use pulse_width::PulseWidthModulation;

use crate::error::KwaversResult;

/// Common trait for all modulation schemes
pub trait Modulation {
    /// Apply modulation to a carrier signal
    fn modulate(&self, carrier: &[f64], t: &[f64]) -> KwaversResult<Vec<f64>>;
    
    /// Demodulate a modulated signal
    fn demodulate(&self, signal: &[f64], t: &[f64]) -> KwaversResult<Vec<f64>>;
}

/// Modulation parameters
#[derive(Debug, Clone)]
pub struct ModulationParams {
    /// Carrier frequency in Hz
    pub carrier_freq: f64,
    /// Sampling rate in Hz
    pub sample_rate: f64,
    /// Modulation index
    pub modulation_index: f64,
}

/// Physical constants for modulation
pub mod constants {
    /// Maximum modulation index for stable AM
    pub const MAX_AM_INDEX: f64 = 1.0;
    
    /// Typical FM deviation for broadcast radio (Hz)
    pub const FM_BROADCAST_DEVIATION: f64 = 75e3;
    
    /// Carson's bandwidth rule factor
    pub const CARSON_BANDWIDTH_FACTOR: f64 = 2.0;
}