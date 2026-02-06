//! Null signal implementation
//!
//! This signal always returns zero amplitude, useful for testing
//! and as a placeholder in composite sources.

use crate::domain::signal::Signal;
use std::fmt::Debug;

/// Null signal that always returns zero
#[derive(Debug, Clone, Copy, Default)]
pub struct NullSignal;

impl NullSignal {
    /// Create a new null signal
    pub fn new() -> Self {
        Self
    }
}

impl Signal for NullSignal {
    fn amplitude(&self, _t: f64) -> f64 {
        0.0
    }

    fn frequency(&self, _t: f64) -> f64 {
        0.0
    }

    fn phase(&self, _t: f64) -> f64 {
        0.0
    }

    fn duration(&self) -> Option<f64> {
        Some(0.0)
    }

    fn clone_box(&self) -> Box<dyn Signal> {
        Box::new(*self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::signal::Signal;

    #[test]
    fn test_null_signal_amplitude() {
        let signal = NullSignal::new();
        assert_eq!(signal.amplitude(0.0), 0.0);
        assert_eq!(signal.amplitude(1.0), 0.0);
        assert_eq!(signal.amplitude(100.0), 0.0);
    }

    #[test]
    fn test_null_signal_frequency() {
        let signal = NullSignal::new();
        assert_eq!(signal.frequency(0.0), 0.0);
        assert_eq!(signal.frequency(1.0), 0.0);
    }

    #[test]
    fn test_null_signal_phase() {
        let signal = NullSignal::new();
        assert_eq!(signal.phase(0.0), 0.0);
        assert_eq!(signal.phase(1.0), 0.0);
    }

    #[test]
    fn test_null_signal_duration() {
        let signal = NullSignal::new();
        assert_eq!(signal.duration(), Some(0.0));
    }

    #[test]
    fn test_null_signal_clone() {
        let signal = NullSignal::new();
        let cloned = signal.clone_box();
        assert_eq!(cloned.amplitude(0.0), 0.0);
    }
}
