//! Physical constants for cavitation detection

/// Subharmonic detection threshold (relative to fundamental)
pub const SUBHARMONIC_THRESHOLD: f64 = 0.1;

/// Broadband noise floor increase for cavitation detection (dB)
pub const BROADBAND_THRESHOLD_DB: f64 = 6.0;

/// Harmonic detection threshold (relative to fundamental)
pub const HARMONIC_THRESHOLD: f64 = 0.05;

/// Minimum spectral power for valid detection
pub const MIN_SPECTRAL_POWER: f64 = 1e-6;

/// Window size for spectral analysis
pub const SPECTRAL_WINDOW_SIZE: usize = 1024;

/// Overlap ratio for spectral windows
pub const WINDOW_OVERLAP_RATIO: f64 = 0.5;

/// Maximum harmonics to analyze
pub const MAX_HARMONICS: usize = 10;

/// Frequency tolerance for peak detection (Hz)
pub const FREQUENCY_TOLERANCE: f64 = 10.0;

/// Minimum SNR for valid detection (dB)
pub const MIN_SNR_DB: f64 = 3.0;

/// Temporal smoothing factor
pub const TEMPORAL_SMOOTHING: f64 = 0.1;

/// Confidence decay rate
pub const CONFIDENCE_DECAY: f64 = 0.95;
