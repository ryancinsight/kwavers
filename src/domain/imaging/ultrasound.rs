//! Ultrasound imaging domain definitions
//!
//! # Nomenclature
//! - B-Mode: Brightness mode (grayscale structure)
//! - Doppler: Flow velocity imaging
//! - Elastography: Tissue stiffness imaging
//! - Harmonic: Nonlinear response imaging

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
    /// Center frequency (Hz)
    pub frequency: f64,
    /// Sampling frequency (Hz)
    pub sampling_frequency: f64,
    /// Dynamic range (dB)
    pub dynamic_range: f64,
    /// Time gain compensation
    pub tgc_enabled: bool,
}

impl Default for UltrasoundConfig {
    fn default() -> Self {
        Self {
            mode: UltrasoundMode::BMode,
            frequency: 5e6,
            sampling_frequency: 40e6,
            dynamic_range: 60.0,
            tgc_enabled: true,
        }
    }
}
