//! Core types for cavitation detection

use std::collections::VecDeque;

/// Detection methods for cavitation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DetectionMethod {
    Subharmonic,   // f0/2, f0/3, etc.
    Ultraharmonic, // 3f0/2, 5f0/2, etc.
    Broadband,     // Increased broadband noise
    Harmonic,      // 2f0, 3f0, etc.
    Combined,      // Combination of methods
}

/// Cavitation state classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CavitationState {
    None,
    Stable,    // Stable cavitation (non-inertial)
    Inertial,  // Inertial cavitation (violent collapse)
    Transient, // Transitioning between states
}

/// Cavitation detection metrics
#[derive(Debug, Clone)]
pub struct CavitationMetrics {
    pub state: CavitationState,
    pub subharmonic_level: f64,
    pub ultraharmonic_level: f64,
    pub broadband_level: f64,
    pub harmonic_distortion: f64,
    pub confidence: f64,
    // Legacy compatibility fields (to be removed when feedback_controller is refactored)
    pub intensity: f64,
    pub harmonic_content: f64,
    pub cavitation_dose: f64,
}

impl Default for CavitationMetrics {
    fn default() -> Self {
        Self {
            state: CavitationState::None,
            subharmonic_level: 0.0,
            ultraharmonic_level: 0.0,
            broadband_level: 0.0,
            harmonic_distortion: 0.0,
            confidence: 0.0,
            intensity: 0.0,
            harmonic_content: 0.0,
            cavitation_dose: 0.0,
        }
    }
}

/// History buffer for temporal analysis
pub struct HistoryBuffer<T> {
    buffer: VecDeque<T>,
    capacity: usize,
}

impl<T: Clone> HistoryBuffer<T> {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, value: T) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(value);
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.buffer.iter()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}
