//! Pulse sequence generation for power modulation

/// Descriptor for a single pulse in a sequence
#[derive(Debug, Clone)]
pub struct PulseDescriptor {
    /// Pulse amplitude (0-1)
    pub amplitude: f64,
    /// Pulse duration (seconds)
    pub duration: f64,
    /// Delay after pulse (seconds)
    pub delay: f64,
    /// Frequency (Hz)
    pub frequency: f64,
}

/// Pulse sequence generator for complex modulation patterns
#[derive(Debug, Clone)]
pub struct PulseSequenceGenerator {
    sequence: Vec<PulseDescriptor>,
    current_index: usize,
    current_time: f64,
    repeat: bool,
}

impl PulseSequenceGenerator {
    /// Create new pulse sequence generator
    pub fn new() -> Self {
        Self {
            sequence: Vec::new(),
            current_index: 0,
            current_time: 0.0,
            repeat: true,
        }
    }

    /// Add pulse to sequence
    pub fn add_pulse(&mut self, pulse: PulseDescriptor) {
        self.sequence.push(pulse);
    }

    /// Create a standard burst sequence
    pub fn create_burst_sequence(
        num_pulses: usize,
        pulse_duration: f64,
        pulse_delay: f64,
        amplitude: f64,
        frequency: f64,
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
    pub fn get_current_pulse(&mut self, dt: f64) -> Option<&PulseDescriptor> {
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
        self.current_time = 0.0;
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
    pub fn total_duration(&self) -> f64 {
        self.sequence.iter().map(|p| p.duration + p.delay).sum()
    }
}
