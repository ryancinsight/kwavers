//! Configuration for harmonic detection

/// Configuration for harmonic detection
#[derive(Debug, Clone)]
pub struct HarmonicDetectionConfig {
    /// Fundamental frequency (Hz)
    pub fundamental_frequency: f64,
    /// Number of harmonics to detect
    pub n_harmonics: usize,
    /// FFT window size
    pub fft_window_size: usize,
    /// Overlap between FFT windows
    pub fft_overlap: f64,
    /// Minimum SNR for harmonic detection (dB)
    pub min_snr_db: f64,
    /// Phase unwrapping enabled
    pub enable_phase_unwrapping: bool,
}

impl Default for HarmonicDetectionConfig {
    fn default() -> Self {
        Self {
            fundamental_frequency: 50.0, // 50 Hz typical for SWE
            n_harmonics: 3,              // Fundamental + 2 harmonics
            fft_window_size: 1024,
            fft_overlap: 0.5, // 50% overlap
            min_snr_db: 10.0, // 10 dB minimum SNR
            enable_phase_unwrapping: true,
        }
    }
}
