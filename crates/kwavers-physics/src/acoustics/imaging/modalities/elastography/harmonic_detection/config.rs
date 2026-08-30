//! Whole-record harmonic-analysis configuration.

/// Selects the harmonics extracted from one complete displacement record.
///
/// Windowing, segmentation, signal-to-noise reporting, and phase conventions
/// are properties of [`super::HarmonicDetector::analyze_harmonics`], not
/// configurable policies.
#[derive(Debug, Clone)]
pub struct HarmonicDetectionConfig {
    /// Fundamental frequency in hertz.
    pub fundamental_frequency: f64,
    /// Total requested harmonic count, including the fundamental.
    pub n_harmonics: usize,
}

impl Default for HarmonicDetectionConfig {
    fn default() -> Self {
        Self {
            fundamental_frequency: 50.0, // 50 Hz typical for SWE
            n_harmonics: 3,              // Fundamental + 2 harmonics
        }
    }
}
