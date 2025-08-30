//! Power Modulation for Cavitation Control
//!
//! Implements amplitude and power modulation schemes for controlling cavitation activity.
//!
//! References:
//! - Hockham et al. (2010): "A real-time controller for sustaining thermally relevant acoustic cavitation"
//! - O'Reilly & Hynynen (2012): "Blood-brain barrier: real-time feedback-controlled focused ultrasound"
//! - Tsai et al. (2016): "Real-time monitoring of focused ultrasound blood-brain barrier opening"

use std::collections::VecDeque;
use std::f64::consts::PI;

// Power modulation constants
/// Default pulse repetition frequency (PRF) in Hz
const DEFAULT_PRF: f64 = 100.0;

/// Default duty cycle (0-1)
const DEFAULT_DUTY_CYCLE: f64 = 0.5;

/// Minimum duty cycle to prevent complete shutdown
const MIN_DUTY_CYCLE: f64 = 0.01;

/// Maximum duty cycle for safety
const MAX_DUTY_CYCLE: f64 = 0.95;

/// Default ramp time for smooth transitions (seconds)
const DEFAULT_RAMP_TIME: f64 = 0.01;

/// Maximum amplitude change rate (per second)
const MAX_AMPLITUDE_RATE: f64 = 10.0;

/// Safety threshold for mechanical index
const MECHANICAL_INDEX_LIMIT: f64 = 1.9;

/// Modulation schemes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModulationScheme {
    Continuous,          // No modulation
    Pulsed,              // On/off pulsing
    DutyCycleControl,    // Variable duty cycle
    AmplitudeModulation, // Continuous amplitude control
    BurstMode,           // Burst sequences
    Ramped,              // Ramped amplitude
    Sinusoidal,          // Sinusoidal modulation
}

/// Power control parameters
#[derive(Debug, Clone)]
pub struct PowerControl {
    pub amplitude: f64,        // Current amplitude (0-1)
    pub duty_cycle: f64,       // Current duty cycle (0-1)
    pub prf: f64,              // Pulse repetition frequency
    pub burst_length: usize,   // Number of cycles per burst
    pub ramp_time: f64,        // Ramp up/down time
    pub mechanical_index: f64, // Current MI
}

impl Default for PowerControl {
    fn default() -> Self {
        Self {
            amplitude: 1.0,
            duty_cycle: DEFAULT_DUTY_CYCLE,
            prf: DEFAULT_PRF,
            burst_length: 10,
            ramp_time: DEFAULT_RAMP_TIME,
            mechanical_index: 0.0,
        }
    }
}

/// Power modulator for controlling ultrasound output
#[derive(Debug)]
pub struct PowerModulator {
    scheme: ModulationScheme,
    control: PowerControl,
    current_phase: f64,
    sample_rate: f64,
    time: f64,
    amplitude_filter: ExponentialFilter,
    safety_limiter: SafetyLimiter,
}

impl PowerModulator {
    pub fn new(scheme: ModulationScheme, sample_rate: f64) -> Self {
        Self {
            scheme,
            control: PowerControl::default(),
            current_phase: 0.0,
            sample_rate,
            time: 0.0,
            amplitude_filter: ExponentialFilter::new(0.01),
            safety_limiter: SafetyLimiter::new(),
        }
    }

    /// Set modulation scheme
    pub fn set_scheme(&mut self, scheme: ModulationScheme) {
        self.scheme = scheme;
    }

    /// Update control parameters
    pub fn update_control(&mut self, control: PowerControl) {
        // Apply safety limits
        self.control = control;
        self.control.duty_cycle = self
            .control
            .duty_cycle
            .clamp(MIN_DUTY_CYCLE, MAX_DUTY_CYCLE);
        self.control.amplitude = self.control.amplitude.clamp(0.0, 1.0);
    }

    /// Get modulation envelope at current time
    pub fn get_envelope(&mut self, dt: f64) -> f64 {
        self.time += dt;

        let envelope = match self.scheme {
            ModulationScheme::Continuous => 1.0,

            ModulationScheme::Pulsed => {
                let period = 1.0 / self.control.prf;
                let phase = (self.time % period) / period;
                if phase < self.control.duty_cycle {
                    1.0
                } else {
                    0.0
                }
            }

            ModulationScheme::DutyCycleControl => {
                let period = 1.0 / self.control.prf;
                let phase = (self.time % period) / period;

                // Smooth transitions using raised cosine
                let transition_width = 0.05; // 5% of period

                if phase < transition_width {
                    // Rising edge
                    0.5 * (1.0 + (PI * phase / transition_width).cos())
                } else if phase < self.control.duty_cycle - transition_width {
                    1.0
                } else if phase < self.control.duty_cycle {
                    // Falling edge
                    let fall_phase =
                        (phase - (self.control.duty_cycle - transition_width)) / transition_width;
                    0.5 * (1.0 - (PI * fall_phase).cos())
                } else {
                    0.0
                }
            }

            ModulationScheme::AmplitudeModulation => {
                // Direct amplitude control with filtering
                self.control.amplitude
            }

            ModulationScheme::BurstMode => {
                let burst_period = 1.0 / self.control.prf;
                let phase = self.time % burst_period;
                let cycles_per_burst = self.control.burst_length as f64;
                let burst_duration = cycles_per_burst / (self.control.prf * 10.0); // Assuming 10x carrier freq

                if phase < burst_duration {
                    1.0
                } else {
                    0.0
                }
            }

            ModulationScheme::Ramped => {
                let period = 1.0 / self.control.prf;
                let phase = (self.time % period) / period;

                if phase < self.control.ramp_time {
                    // Ramp up
                    phase / self.control.ramp_time
                } else if phase < self.control.duty_cycle - self.control.ramp_time {
                    1.0
                } else if phase < self.control.duty_cycle {
                    // Ramp down
                    let ramp_phase = (self.control.duty_cycle - phase) / self.control.ramp_time;
                    ramp_phase.max(0.0)
                } else {
                    0.0
                }
            }

            ModulationScheme::Sinusoidal => {
                let modulation_freq = self.control.prf;
                0.5 * (1.0 + (2.0 * PI * modulation_freq * self.time).sin())
            }
        };

        // Apply amplitude control and filtering
        let modulated = envelope * self.control.amplitude;
        let filtered = self.amplitude_filter.filter(modulated);

        // Apply safety limiting
        self.safety_limiter
            .limit(filtered, self.control.mechanical_index)
    }

    /// Reset modulator state
    pub fn reset(&mut self) {
        self.time = 0.0;
        self.current_phase = 0.0;
        self.amplitude_filter.reset();
    }

    /// Get current control parameters
    pub fn control(&self) -> &PowerControl {
        &self.control
    }
}

/// Amplitude controller with smooth transitions
#[derive(Debug)]
pub struct AmplitudeController {
    target_amplitude: f64,
    current_amplitude: f64,
    rate_limit: f64,
    filter: ExponentialFilter,
}

impl AmplitudeController {
    pub fn new(initial_amplitude: f64) -> Self {
        Self {
            target_amplitude: initial_amplitude,
            current_amplitude: initial_amplitude,
            rate_limit: MAX_AMPLITUDE_RATE,
            filter: ExponentialFilter::new(0.01),
        }
    }

    /// Set target amplitude
    pub fn set_target(&mut self, target: f64) {
        self.target_amplitude = target.clamp(0.0, 1.0);
    }

    /// Update amplitude with rate limiting
    pub fn update(&mut self, dt: f64) -> f64 {
        let max_change = self.rate_limit * dt;
        let error = self.target_amplitude - self.current_amplitude;

        let change = if error.abs() > max_change {
            max_change * error.signum()
        } else {
            error
        };

        self.current_amplitude += change;
        self.current_amplitude = self.filter.filter(self.current_amplitude);

        self.current_amplitude
    }

    /// Get current amplitude
    pub fn amplitude(&self) -> f64 {
        self.current_amplitude
    }
}

/// Duty cycle controller for pulsed operation
#[derive(Debug)]
pub struct DutyCycleController {
    target_duty_cycle: f64,
    current_duty_cycle: f64,
    min_duty: f64,
    max_duty: f64,
    transition_rate: f64,
}

impl DutyCycleController {
    pub fn new(initial_duty_cycle: f64) -> Self {
        Self {
            target_duty_cycle: initial_duty_cycle,
            current_duty_cycle: initial_duty_cycle,
            min_duty: MIN_DUTY_CYCLE,
            max_duty: MAX_DUTY_CYCLE,
            transition_rate: 1.0, // Full range in 1 second
        }
    }

    /// Set target duty cycle
    pub fn set_target(&mut self, target: f64) {
        self.target_duty_cycle = target.clamp(self.min_duty, self.max_duty);
    }

    /// Update duty cycle with smooth transition
    pub fn update(&mut self, dt: f64) -> f64 {
        let max_change = self.transition_rate * dt;
        let error = self.target_duty_cycle - self.current_duty_cycle;

        let change = if error.abs() > max_change {
            max_change * error.signum()
        } else {
            error
        };

        self.current_duty_cycle += change;
        self.current_duty_cycle = self.current_duty_cycle.clamp(self.min_duty, self.max_duty);

        self.current_duty_cycle
    }

    /// Get current duty cycle
    pub fn duty_cycle(&self) -> f64 {
        self.current_duty_cycle
    }

    /// Set duty cycle limits
    pub fn set_limits(&mut self, min: f64, max: f64) {
        self.min_duty = min.max(MIN_DUTY_CYCLE);
        self.max_duty = max.min(MAX_DUTY_CYCLE);
    }
}

/// Exponential filter for smooth transitions
struct ExponentialFilter {
    alpha: f64,
    state: f64,
}

impl ExponentialFilter {
    fn new(time_constant: f64) -> Self {
        Self {
            alpha: (-1.0 / time_constant).exp(),
            state: 0.0,
        }
    }

    fn filter(&mut self, input: f64) -> f64 {
        self.state = self.alpha * self.state + (1.0 - self.alpha) * input;
        self.state
    }

    fn reset(&mut self) {
        self.state = 0.0;
    }
}

/// Safety limiter to prevent excessive power
struct SafetyLimiter {
    max_power: f64,
    max_mi: f64,
    history: VecDeque<f64>,
}

impl SafetyLimiter {
    fn new() -> Self {
        Self {
            max_power: 1.0,
            max_mi: MECHANICAL_INDEX_LIMIT,
            history: VecDeque::with_capacity(100),
        }
    }

    fn limit(&mut self, amplitude: f64, current_mi: f64) -> f64 {
        // Limit based on mechanical index
        let mi_limit = if current_mi > 0.0 {
            amplitude * (self.max_mi / current_mi).min(1.0)
        } else {
            amplitude
        };

        // Track power history
        self.history.push_back(amplitude * amplitude);
        if self.history.len() > 100 {
            self.history.pop_front();
        }

        // Calculate average power
        let avg_power: f64 = self.history.iter().sum::<f64>() / self.history.len() as f64;

        // Limit if average power exceeds threshold
        if avg_power > self.max_power {
            mi_limit * (self.max_power / avg_power).sqrt()
        } else {
            mi_limit
        }
    }
}

/// Pulse sequence generator for complex modulation patterns
#[derive(Debug)]
pub struct PulseSequenceGenerator {
    sequence: Vec<PulseDescriptor>,
    current_index: usize,
    current_time: f64,
    repeat: bool,
}

#[derive(Debug, Clone)]
pub struct PulseDescriptor {
    pub amplitude: f64,
    pub duration: f64,
    pub frequency: f64,
    pub phase: f64,
}

impl PulseSequenceGenerator {
    pub fn new(sequence: Vec<PulseDescriptor>, repeat: bool) -> Self {
        Self {
            sequence,
            current_index: 0,
            current_time: 0.0,
            repeat,
        }
    }

    /// Get current pulse parameters
    pub fn update(&mut self, dt: f64) -> Option<PulseDescriptor> {
        if self.sequence.is_empty() || (!self.repeat && self.current_index >= self.sequence.len()) {
            return None;
        }

        self.current_time += dt;

        let current_pulse = &self.sequence[self.current_index];

        if self.current_time >= current_pulse.duration {
            self.current_time = 0.0;
            self.current_index += 1;

            if self.repeat && self.current_index >= self.sequence.len() {
                self.current_index = 0;
            }
        }

        if self.current_index < self.sequence.len() {
            Some(self.sequence[self.current_index].clone())
        } else {
            None
        }
    }

    /// Reset sequence to beginning
    pub fn reset(&mut self) {
        self.current_index = 0;
        self.current_time = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pulsed_modulation() {
        let mut modulator = PowerModulator::new(ModulationScheme::Pulsed, 1000.0);
        modulator.update_control(PowerControl {
            duty_cycle: 0.3,
            prf: 100.0,
            ..Default::default()
        });

        // Sample over one period
        let period = 1.0 / 100.0;
        let dt = period / 100.0;

        let mut on_time = 0.0;
        for _ in 0..100 {
            let envelope = modulator.get_envelope(dt);
            if envelope > 0.5 {
                on_time += dt;
            }
        }

        // Should be approximately 30% duty cycle
        let actual_duty = on_time / period;
        assert!((actual_duty - 0.3).abs() < 0.05);
    }

    #[test]
    fn test_amplitude_controller() {
        let mut controller = AmplitudeController::new(0.0);
        controller.set_target(1.0);

        // Update with rate limiting
        let final_amplitude = controller.update(0.1);

        // Should not jump instantly to 1.0 due to rate limiting
        assert!(final_amplitude < 1.0);
        assert!(final_amplitude > 0.0);
    }

    #[test]
    fn test_duty_cycle_limits() {
        let mut controller = DutyCycleController::new(0.5);
        controller.set_limits(0.2, 0.8);

        // Try to set outside limits
        controller.set_target(0.1);
        controller.update(1.0);
        assert!(controller.duty_cycle() >= 0.2);

        controller.set_target(0.9);
        controller.update(1.0);
        assert!(controller.duty_cycle() <= 0.8);
    }
}
