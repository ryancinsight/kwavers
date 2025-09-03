//! Cavitation detection using ML techniques

/// Cavitation detector using machine learning
#[derive(Debug)]
pub struct CavitationDetector {
    threshold: f64,
    sensitivity: f64,
}

/// Detected cavitation event
#[derive(Debug, Clone)]
pub struct CavitationEvent {
    pub time: f64,
    pub position: [f64; 3],
    pub intensity: f64,
    pub event_type: CavitationEventType,
}

/// Type of cavitation event
#[derive(Debug, Clone, Copy)]
pub enum CavitationEventType {
    Inception,
    StableOscillation,
    InertialCollapse,
    Rebound,
}

impl CavitationDetector {
    /// Create a new cavitation detector
    #[must_use]
    pub fn new(threshold: f64, sensitivity: f64) -> Self {
        Self {
            threshold,
            sensitivity,
        }
    }
}
