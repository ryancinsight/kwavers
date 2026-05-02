use crate::core::constants::SOUND_SPEED_TISSUE;
use serde::{Deserialize, Serialize};

/// Apodization window types for sidelobe suppression.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ApodizationType {
    /// No apodization (uniform weighting).
    None,
    /// Hamming window.
    Hamming,
    /// Hanning window.
    Hanning,
    /// Blackman window.
    Blackman,
}

/// Configuration for delay-and-sum PAM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelayAndSumConfig {
    /// Sound speed in medium (m/s).
    pub sound_speed: f64,
    /// Sampling frequency (Hz).
    pub sampling_frequency: f64,
    /// Detection threshold (multiple of noise floor).
    pub detection_threshold: f64,
    /// Temporal window size (samples).
    pub window_size: usize,
    /// Apodization window type.
    pub apodization: ApodizationType,
    /// Enable coherence factor weighting.
    pub coherence_weighting: bool,
}

impl Default for DelayAndSumConfig {
    fn default() -> Self {
        Self {
            sound_speed: SOUND_SPEED_TISSUE,
            sampling_frequency: 5e6,
            detection_threshold: 3.0,
            window_size: 512,
            apodization: ApodizationType::Hamming,
            coherence_weighting: true,
        }
    }
}

/// Detected cavitation event.
#[derive(Debug, Clone)]
pub struct CavitationEvent {
    /// 3D position (m).
    pub position: [f64; 3],
    /// Intensity (arbitrary units).
    pub intensity: f64,
    /// Time of occurrence (s).
    pub time: f64,
    /// Coherence factor (0–1).
    pub coherence: f64,
    /// Peak frequency content (Hz); `None` if not estimated.
    pub peak_frequency: Option<f64>,
}
