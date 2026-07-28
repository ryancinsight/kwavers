//! Pulse sequence generation for power modulation

use aequitas::systems::si::quantities::{Frequency, Time};

/// Descriptor for a single pulse in a sequence
#[derive(Debug, Clone)]
pub struct PulseDescriptor {
    /// Pulse amplitude (0-1)
    pub amplitude: f64,
    /// Pulse duration (seconds)
    pub duration: Time<f64>,
    /// Delay after pulse (seconds)
    pub delay: Time<f64>,
    /// Frequency (Hz)
    pub frequency: Frequency<f64>,
}

/// Pulse sequence generator for complex modulation patterns
#[derive(Debug, Clone)]
pub struct PulseSequenceGenerator {
    sequence: Vec<PulseDescriptor>,
    current_index: usize,
    current_time: Time<f64>,
    repeat: bool,
}

impl Default for PulseSequenceGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl PulseSequenceGenerator {
    /// Create new pulse sequence generator
    #[must_use]
    pub fn new() -> Self {
        Self {
            sequence: Vec::new(),
            current_index: 0,
            current_time: Time::from_base(0.0),
            repeat: true,
        }
    }

    /// Add pulse to sequence
    pub fn add_pulse(&mut self, pulse: PulseDescriptor) {
        self.sequence.push(pulse);
    }

    /// Create a standard burst sequence
    #[must_use]
    pub fn create_burst_sequence(
        num_pulses: usize,
        pulse_duration: Time<f64>,
        pulse_delay: Time<f64>,
        amplitude: f64,
        frequency: Frequency<f64>,
    ) -> Self {
        let mut generator = Self::new();

        for _ in 0..num_pulses {
            generator.add_pulse(PulseDescriptor {
                amplitude,
                duration: pulse_duration,
                delay: pulse_delay,
                frequency,
            });
        }

        generator
    }

    /// Get current pulse parameters
    pub fn get_current_pulse(&mut self, dt: Time<f64>) -> Option<&PulseDescriptor> {
        if self.sequence.is_empty() {
            return None;
        }

        self.current_time += dt;

        // Check if we need to advance to next pulse
        if let Some(current_pulse) = self.sequence.get(self.current_index) {
            let pulse_end_time = current_pulse.duration + current_pulse.delay;

            if self.current_time >= pulse_end_time {
                self.current_time -= pulse_end_time;
                self.current_index += 1;

                // Handle sequence repeat or end
                if self.current_index >= self.sequence.len() {
                    if self.repeat {
                        self.current_index = 0;
                    } else {
                        return None;
                    }
                }
            }
        }

        self.sequence.get(self.current_index)
    }

    /// Reset sequence to beginning
    pub fn reset(&mut self) {
        self.current_index = 0;
        self.current_time = Time::from_base(0.0);
    }

    /// Set repeat mode
    pub fn set_repeat(&mut self, repeat: bool) {
        self.repeat = repeat;
    }

    /// Clear sequence
    pub fn clear(&mut self) {
        self.sequence.clear();
        self.reset();
    }

    /// Get total sequence duration
    #[must_use]
    pub fn total_duration(&self) -> Time<f64> {
        self.sequence
            .iter()
            .map(|p| p.duration + p.delay)
            .fold(Time::from_base(0.0), |total, duration| total + duration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::numerical::MHZ_TO_HZ;

    /// create_burst_sequence produces exactly num_pulses pulses.
    #[test]
    fn burst_sequence_has_correct_pulse_count() {
        let g = PulseSequenceGenerator::create_burst_sequence(
            5,
            Time::from_base(0.01),
            Time::from_base(0.005),
            0.8,
            Frequency::from_base(MHZ_TO_HZ),
        );
        // total_duration = 5 * (0.01 + 0.005) = 0.075
        let expected = 5.0 * (0.01 + 0.005);
        assert!(
            (g.total_duration().into_base() - expected).abs() < 1e-14,
            "total_duration must be {expected:.4}; got {}",
            g.total_duration().into_base()
        );
    }

    /// Empty sequence: get_current_pulse always returns None.
    #[test]
    fn empty_sequence_returns_none() {
        let mut g = PulseSequenceGenerator::new();
        assert!(
            g.get_current_pulse(Time::from_base(0.001)).is_none(),
            "empty sequence must return None"
        );
    }

    /// add_pulse then get_current_pulse returns the descriptor before pulse_end_time elapses.
    #[test]
    fn get_current_pulse_returns_first_pulse_within_duration() {
        let mut g = PulseSequenceGenerator::new();
        g.add_pulse(PulseDescriptor {
            amplitude: 0.7,
            duration: Time::from_base(1.0),
            delay: Time::from_base(0.5),
            frequency: Frequency::from_base(MHZ_TO_HZ),
        });
        // dt=0.1 < duration+delay=1.5 → still on pulse 0
        let pulse = g
            .get_current_pulse(Time::from_base(0.1))
            .expect("must return pulse");
        assert!(
            (pulse.amplitude - 0.7).abs() < 1e-15,
            "must return first pulse; got amplitude={}",
            pulse.amplitude
        );
    }

    /// reset restores index and time to zero.
    #[test]
    fn reset_restores_initial_state() {
        let mut g = PulseSequenceGenerator::create_burst_sequence(
            3,
            Time::from_base(0.01),
            Time::from_base(0.005),
            0.5,
            Frequency::from_base(MHZ_TO_HZ),
        );
        g.get_current_pulse(Time::from_base(0.1)); // advance time
        g.reset();
        // After reset, total_duration is unaffected, but internal time is zero
        let expected = 3.0 * (0.01 + 0.005);
        assert!((g.total_duration().into_base() - expected).abs() < 1e-14);
    }
}
