//! Time-varying signal implementation
//!
//! This signal uses pre-computed amplitude values, useful for
//! arbitrary waveform generation and experimental signal patterns.

use crate::signal::Signal;
use std::fmt::Debug;

/// Time-varying signal with pre-computed values
#[derive(Debug, Clone)]
pub struct TimeVaryingSignal {
    values: Vec<f64>,
    dt: f64,
    base_frequency: f64,
}

impl TimeVaryingSignal {
    /// Create a new time-varying signal from pre-computed values
    ///
    /// # Arguments
    ///
    /// * `values` - Pre-computed amplitude values
    /// * `dt` - Time step between values
    ///
    /// # Returns
    ///
    /// A new `TimeVaryingSignal` instance
    pub fn new(values: Vec<f64>, dt: f64) -> Self {
        // Estimate base frequency from signal using zero-crossing
        let base_frequency = if dt > 0.0 && !values.is_empty() {
            let zero_crossings = values.windows(2).filter(|w| w[0] * w[1] < 0.0).count() as f64;
            if zero_crossings > 0.0 {
                zero_crossings / (2.0 * dt * values.len() as f64)
            } else {
                1.0 / (dt * values.len() as f64)
            }
        } else {
            0.0
        };

        Self {
            values,
            dt,
            base_frequency,
        }
    }

    /// Get the time step
    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Get the base frequency
    pub fn base_frequency(&self) -> f64 {
        self.base_frequency
    }

    /// Get the signal duration
    pub fn signal_duration(&self) -> f64 {
        self.dt * self.values.len() as f64
    }

    /// Get the number of samples
    pub fn num_samples(&self) -> usize {
        self.values.len()
    }

    /// Get a reference to the signal values
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

impl Signal for TimeVaryingSignal {
    fn amplitude(&self, t: f64) -> f64 {
        if self.dt <= 0.0 || self.values.is_empty() {
            return 0.0;
        }

        let index = (t / self.dt) as usize;
        if index < self.values.len() {
            self.values[index]
        } else {
            0.0
        }
    }

    fn frequency(&self, _t: f64) -> f64 {
        self.base_frequency
    }

    fn phase(&self, t: f64) -> f64 {
        // Phase estimation based on time index
        2.0 * std::f64::consts::PI * self.base_frequency * t
    }

    fn duration(&self) -> Option<f64> {
        if self.dt > 0.0 && !self.values.is_empty() {
            Some(self.dt * self.values.len() as f64)
        } else {
            None
        }
    }

    fn clone_box(&self) -> Box<dyn Signal> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::Signal;

    #[test]
    fn test_time_varying_signal_creation() {
        let values = vec![0.0, 1.0, 0.0, -1.0, 0.0];
        let dt = 1e-6;
        let signal = TimeVaryingSignal::new(values.clone(), dt);

        assert_eq!(signal.dt(), dt);
        assert_eq!(signal.num_samples(), 5);
        assert_eq!(signal.values(), &values);
    }

    #[test]
    fn test_time_varying_signal_amplitude() {
        let values = vec![0.0, 1.0, 0.5, 0.0, -0.5];
        let dt = 1e-6;
        let signal = TimeVaryingSignal::new(values, dt);

        assert_eq!(signal.amplitude(0.0), 0.0);
        assert_eq!(signal.amplitude(1e-6), 1.0);
        assert_eq!(signal.amplitude(2e-6), 0.5);
        assert_eq!(signal.amplitude(4e-6), -0.5);
        assert_eq!(signal.amplitude(10e-6), 0.0); // Beyond signal duration
    }

    #[test]
    fn test_time_varying_signal_frequency() {
        let values = vec![0.0, 1.0, 0.0, -1.0, 0.0];
        let dt = 1e-6;
        let signal = TimeVaryingSignal::new(values, dt);

        assert!(signal.frequency(0.0) > 0.0);
        assert_eq!(signal.frequency(1.0), signal.base_frequency());
    }

    #[test]
    fn test_time_varying_signal_duration() {
        let values = vec![0.0, 1.0, 0.0, -1.0];
        let dt = 1e-6;
        let signal = TimeVaryingSignal::new(values, dt);

        assert_eq!(signal.duration(), Some(4e-6));
    }

    #[test]
    fn test_time_varying_signal_empty() {
        let values = vec![];
        let dt = 1e-6;
        let signal = TimeVaryingSignal::new(values, dt);

        assert_eq!(signal.amplitude(0.0), 0.0);
        assert_eq!(signal.duration(), None);
    }

    #[test]
    fn test_time_varying_signal_clone() {
        let values = vec![0.0, 1.0, 0.0];
        let dt = 1e-6;
        let signal = TimeVaryingSignal::new(values, dt);
        let cloned = signal.clone_box();

        assert_eq!(cloned.amplitude(0.0), 0.0);
        assert_eq!(cloned.amplitude(1e-6), 1.0);
    }
}
