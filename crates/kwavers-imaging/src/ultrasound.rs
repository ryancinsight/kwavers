//! Ultrasound imaging domain definitions
//!
//! # Nomenclature
//! - B-Mode: Brightness mode (grayscale structure)
//! - Doppler: Flow velocity imaging
//! - Elastography: Tissue stiffness imaging
//! - Harmonic: Nonlinear response imaging

use aequitas::systems::si::quantities::Frequency;

pub mod ceus;
pub mod elastography;
pub mod hifu;

/// Ultrasound imaging mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UltrasoundMode {
    /// Brightness mode (grayscale)
    BMode,
    /// Doppler flow imaging
    Doppler,
    /// Tissue elasticity imaging
    Elastography,
    /// Harmonic imaging
    Harmonic,
}

/// Ultrasound imaging configuration
#[derive(Debug, Clone)]
pub struct UltrasoundConfig {
    /// Imaging mode
    pub mode: UltrasoundMode,
    /// Center frequency.
    pub frequency: Frequency,
    /// Sampling frequency.
    pub sampling_frequency: Frequency,
    /// Dynamic range (dB)
    pub dynamic_range: f64,
    /// Time gain compensation
    pub tgc_enabled: bool,
}

impl Default for UltrasoundConfig {
    fn default() -> Self {
        use kwavers_core::constants::numerical::MHZ_TO_HZ;
        Self {
            mode: UltrasoundMode::BMode,
            frequency: Frequency::from_base(5.0 * MHZ_TO_HZ),
            sampling_frequency: Frequency::from_base(40.0 * MHZ_TO_HZ),
            dynamic_range: 60.0,
            tgc_enabled: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_ultrasound_frequencies_are_typed() {
        let config = UltrasoundConfig::default();
        assert_eq!(config.frequency.into_base(), 5.0e6);
        assert_eq!(config.sampling_frequency.into_base(), 40.0e6);
        assert_eq!(config.dynamic_range, 60.0);
    }
}
