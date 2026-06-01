//! Modulation schemes and power control structures

use super::constants::{
    DEFAULT_DUTY_CYCLE, DEFAULT_PRF, DEFAULT_RAMP_TIME, MAX_DUTY_CYCLE, MIN_DUTY_CYCLE,
};
use crate::core::constants::numerical::TWO_PI;

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
                let omega = TWO_PI * self.prf;
                self.amplitude * 0.5f64.mul_add((omega * time).sin(), 0.5)
            }
            _ => self.amplitude, // Default for other schemes
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::constants::{DEFAULT_DUTY_CYCLE, DEFAULT_PRF};
    use super::*;

    /// ModulationScheme variants are pairwise distinct.
    #[test]
    fn modulation_scheme_variants_distinct() {
        assert_ne!(ModulationScheme::Continuous, ModulationScheme::Pulsed);
        assert_ne!(ModulationScheme::Sinusoidal, ModulationScheme::BurstMode);
    }

    /// PowerControl::default has amplitude=1.0 and duty_cycle=DEFAULT_DUTY_CYCLE.
    #[test]
    fn power_control_default_fields() {
        let pc = PowerControl::default();
        assert!((pc.amplitude - 1.0).abs() < 1e-15);
        assert!((pc.duty_cycle - DEFAULT_DUTY_CYCLE).abs() < 1e-15);
        assert!((pc.prf - DEFAULT_PRF).abs() < 1e-15);
        assert_eq!(pc.scheme, ModulationScheme::Continuous);
    }

    /// instantaneous_power for Continuous returns amplitude at any time.
    #[test]
    fn instantaneous_power_continuous_returns_amplitude() {
        let pc = PowerControl::default(); // Continuous, amplitude=1.0
        for t in [0.0, 0.001, 1.0, 100.0] {
            let p = pc.instantaneous_power(t);
            assert!(
                (p - 1.0).abs() < 1e-15,
                "Continuous power must equal amplitude=1.0 at t={t}"
            );
        }
    }

    /// instantaneous_power for Pulsed is amplitude during ON phase, 0 during OFF.
    ///
    /// prf=10 Hz → period=0.1 s. duty_cycle=0.5.
    /// t=0.04: phase=0.4 < 0.5 → ON; t=0.09: phase=0.9 ≥ 0.5 → OFF.
    #[test]
    fn instantaneous_power_pulsed_on_off_correct() {
        let mut pc = PowerControl::new(ModulationScheme::Pulsed);
        pc.amplitude = 0.8;
        pc.prf = 10.0;
        pc.duty_cycle = 0.5;
        let on_power = pc.instantaneous_power(0.04);
        let off_power = pc.instantaneous_power(0.09);
        assert!(
            (on_power - 0.8).abs() < 1e-14,
            "ON phase must return amplitude=0.8; got {on_power}"
        );
        assert!(
            off_power.abs() < 1e-14,
            "OFF phase must return 0.0; got {off_power}"
        );
    }
}
