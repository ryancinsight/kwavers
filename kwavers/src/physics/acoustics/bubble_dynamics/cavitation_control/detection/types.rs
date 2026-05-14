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
    // Compatibility fields used by the current feedback controller.
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
#[derive(Debug)]
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

#[cfg(test)]
mod tests {
    use super::*;

    /// CavitationMetrics::default has zero-valued numeric fields and CavitationState::None.
    #[test]
    fn default_metrics_are_zero() {
        let m = CavitationMetrics::default();
        assert_eq!(m.state, CavitationState::None);
        assert!((m.subharmonic_level).abs() < 1e-30);
        assert!((m.confidence).abs() < 1e-30);
    }

    /// DetectionMethod variants are pairwise distinct.
    #[test]
    fn detection_method_variants_distinct() {
        assert_ne!(DetectionMethod::Subharmonic, DetectionMethod::Broadband);
        assert_ne!(DetectionMethod::Harmonic, DetectionMethod::Combined);
    }

    /// CavitationState variants are pairwise distinct.
    #[test]
    fn cavitation_state_variants_distinct() {
        assert_ne!(CavitationState::None, CavitationState::Stable);
        assert_ne!(CavitationState::Inertial, CavitationState::Transient);
    }

    /// HistoryBuffer respects capacity: oldest elements are evicted when full.
    #[test]
    fn history_buffer_evicts_oldest_at_capacity() {
        let mut buf: HistoryBuffer<i32> = HistoryBuffer::new(3);
        buf.push(1);
        buf.push(2);
        buf.push(3);
        assert_eq!(buf.len(), 3);
        buf.push(4); // evicts 1
        assert_eq!(buf.len(), 3);
        let values: Vec<i32> = buf.iter().copied().collect();
        assert_eq!(
            values,
            vec![2, 3, 4],
            "oldest element must be evicted: {values:?}"
        );
    }

    /// HistoryBuffer::is_empty() is true for new buffer and false after push.
    #[test]
    fn history_buffer_empty_state_correct() {
        let mut buf: HistoryBuffer<f64> = HistoryBuffer::new(5);
        assert!(buf.is_empty(), "new buffer must be empty");
        buf.push(1.0);
        assert!(!buf.is_empty(), "buffer must not be empty after push");
    }
}
