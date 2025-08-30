//! Acoustic event analysis using ML

/// Acoustic event analyzer
#[derive(Debug))]
pub struct AcousticEventAnalyzer {
    frequency_threshold: f64,
    amplitude_threshold: f64,
}

/// Detected acoustic event
#[derive(Debug, Clone))]
pub struct AcousticEvent {
    pub time: f64,
    pub frequency: f64,
    pub amplitude: f64,
    pub event_type: AcousticEventType,
}

/// Type of acoustic event
#[derive(Debug, Clone, Copy))]
pub enum AcousticEventType {
    Harmonic,
    Subharmonic,
    Broadband,
    ShockWave,
}

impl AcousticEventAnalyzer {
    /// Create a new acoustic event analyzer
    pub fn new(frequency_threshold: f64, amplitude_threshold: f64) -> Self {
        Self {
            frequency_threshold,
            amplitude_threshold,
        }
    }
}
