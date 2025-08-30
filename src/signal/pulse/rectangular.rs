//! Rectangular pulse signal implementation

use crate::signal::Signal;
use std::f64::consts::PI;

use super::{DEFAULT_RISE_TIME_FRACTION, MIN_PULSE_WIDTH};

/// Rectangular pulse with configurable rise/fall times
///
/// Implements a rectangular pulse with smooth transitions:
/// - Rise time: transition from 0 to full amplitude
/// - Fall time: transition from full amplitude to 0
#[derive(Debug, Clone))]
pub struct RectangularPulse {
    center_frequency: f64,
    start_time: f64,
    pulse_width: f64,
    amplitude: f64,
    phase: f64,
    rise_time: f64,
    fall_time: f64,
}

impl RectangularPulse {
    pub fn new(center_frequency: f64, start_time: f64, pulse_width: f64, amplitude: f64) -> Self {
        assert!(center_frequency > 0.0, "Center frequency must be positive");
        assert!(pulse_width >= MIN_PULSE_WIDTH, "Pulse width too small");
        assert!(amplitude >= 0.0, "Amplitude must be non-negative");

        let transition_time = pulse_width * DEFAULT_RISE_TIME_FRACTION;

        Self {
            center_frequency,
            start_time,
            pulse_width,
            amplitude,
            phase: 0.0,
            rise_time: transition_time,
            fall_time: transition_time,
        }
    }

    pub fn with_rise_fall_times(mut self, rise_time: f64, fall_time: f64) -> Self {
        assert!(rise_time >= 0.0 && rise_time < self.pulse_width / 2.0);
        assert!(fall_time >= 0.0 && fall_time < self.pulse_width / 2.0);
        self.rise_time = rise_time;
        self.fall_time = fall_time;
        self
    }

    fn envelope(&self, t: f64) -> f64 {
        let relative_time = t - self.start_time;

        if relative_time < 0.0 {
            0.0
        } else if relative_time < self.rise_time {
            // Smooth rise using raised cosine
            0.5 * (1.0 - (PI * (1.0 - relative_time / self.rise_time)).cos())
        } else if relative_time < self.pulse_width - self.fall_time {
            1.0
        } else if relative_time < self.pulse_width {
            // Smooth fall using raised cosine
            let fall_phase = (relative_time - (self.pulse_width - self.fall_time)) / self.fall_time;
            0.5 * (1.0 + (PI * fall_phase).cos())
        } else {
            0.0
        }
    }
}

impl Signal for RectangularPulse {
    fn amplitude(&self, t: f64) -> f64 {
        let envelope = self.envelope(t);
        let carrier = (2.0 * PI * self.center_frequency * t + self.phase).sin();
        self.amplitude * envelope * carrier
    }

    fn frequency(&self, _t: f64) -> f64 {
        self.center_frequency
    }

    fn phase(&self, _t: f64) -> f64 {
        self.phase
    }

    fn duration(&self) -> Option<f64> {
        Some(self.pulse_width)
    }

    fn clone_box(&self) -> Box<dyn Signal> {
        Box::new(self.clone())
    }
}
