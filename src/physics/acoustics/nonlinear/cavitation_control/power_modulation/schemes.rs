//! Modulation schemes and power control structures

use super::constants::{
    DEFAULT_DUTY_CYCLE, DEFAULT_PRF, DEFAULT_RAMP_TIME, MAX_DUTY_CYCLE, MIN_DUTY_CYCLE,
};

/// Modulation schemes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModulationScheme {
    /// No modulation - continuous wave
    Continuous,
    /// On/off pulsing
    Pulsed,
    /// Variable duty cycle control
    DutyCycleControl,
    /// Continuous amplitude control
    AmplitudeModulation,
    /// Burst sequences
    BurstMode,
    /// Ramped amplitude
    Ramped,
    /// Sinusoidal modulation
    Sinusoidal,
}

/// Power control parameters
#[derive(Debug, Clone)]
pub struct PowerControl {
    /// Current amplitude (0-1)
    pub amplitude: f64,
    /// Duty cycle (0-1)
    pub duty_cycle: f64,
    /// Pulse repetition frequency (Hz)
    pub prf: f64,
    /// Modulation scheme
    pub scheme: ModulationScheme,
    /// Ramp time for transitions (seconds)
    pub ramp_time: f64,
}

impl Default for PowerControl {
    fn default() -> Self {
        Self {
            amplitude: 1.0,
            duty_cycle: DEFAULT_DUTY_CYCLE,
            prf: DEFAULT_PRF,
            scheme: ModulationScheme::Continuous,
            ramp_time: DEFAULT_RAMP_TIME,
        }
    }
}

impl PowerControl {
    /// Create new power control with specified scheme
    #[must_use]
    pub fn new(scheme: ModulationScheme) -> Self {
        Self {
            scheme,
            ..Default::default()
        }
    }

    /// Set amplitude with safety limits
    pub fn set_amplitude(&mut self, amplitude: f64) {
        self.amplitude = amplitude.clamp(0.0, 1.0);
    }

    /// Set duty cycle with safety limits
    pub fn set_duty_cycle(&mut self, duty_cycle: f64) {
        self.duty_cycle = duty_cycle.clamp(MIN_DUTY_CYCLE, MAX_DUTY_CYCLE);
    }

    /// Calculate instantaneous power
    #[must_use]
    pub fn instantaneous_power(&self, time: f64) -> f64 {
        match self.scheme {
            ModulationScheme::Continuous => self.amplitude,
            ModulationScheme::Pulsed => {
                let period = 1.0 / self.prf;
                let phase = (time % period) / period;
                if phase < self.duty_cycle {
                    self.amplitude
                } else {
                    0.0
                }
            }
            ModulationScheme::Sinusoidal => {
                let omega = 2.0 * std::f64::consts::PI * self.prf;
                self.amplitude * (0.5 + 0.5 * (omega * time).sin())
            }
            _ => self.amplitude, // Default for other schemes
        }
    }
}
