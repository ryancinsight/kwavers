//! Main power modulator implementation

use super::constants::*;
use super::filters::ExponentialFilter;
use super::safety::SafetyLimiter;
use super::schemes::{ModulationScheme, PowerControl};
use std::f64::consts::PI;

/// Power modulator for controlling ultrasound output
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
    /// Create new power modulator
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
                
                // Smooth transition using cosine taper
                let transition_width = 0.05; // 5% of period for transition
                
                if phase < self.control.duty_cycle - transition_width {
                    1.0
                } else if phase < self.control.duty_cycle {
                    let t = (phase - (self.control.duty_cycle - transition_width)) / transition_width;
                    0.5 * (1.0 + (PI * t).cos())
                } else if phase < self.control.duty_cycle + transition_width {
                    let t = (phase - self.control.duty_cycle) / transition_width;
                    0.5 * (1.0 - (PI * t).cos())
                } else {
                    0.0
                }
            }

            ModulationScheme::AmplitudeModulation => self.control.amplitude,

            ModulationScheme::BurstMode => {
                // Burst mode: groups of pulses
                let burst_period = 0.1; // 10 Hz burst rate
                let burst_duration = 0.02; // 20ms bursts
                let burst_phase = (self.time % burst_period) / burst_period;
                
                if burst_phase < burst_duration / burst_period {
                    let pulse_phase = (self.time * self.control.prf) % 1.0;
                    if pulse_phase < self.control.duty_cycle {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            }

            ModulationScheme::Ramped => {
                // Linear ramp up and down
                let ramp_duration = self.control.ramp_time;
                
                if self.time < ramp_duration {
                    self.time / ramp_duration
                } else {
                    1.0
                }
            }

            ModulationScheme::Sinusoidal => {
                let omega = 2.0 * PI * self.control.prf;
                0.5 * (1.0 + (omega * self.time).sin())
            }
        };

        // Apply amplitude control and filtering
        let scaled = envelope * self.control.amplitude;
        let filtered = self.amplitude_filter.filter(scaled);
        
        // Apply safety limiting
        self.safety_limiter.limit(filtered)
    }

    /// Reset modulator state
    pub fn reset(&mut self) {
        self.time = 0.0;
        self.current_phase = 0.0;
        self.amplitude_filter.reset();
    }

    /// Get current time
    pub fn get_time(&self) -> f64 {
        self.time
    }

    /// Get effective duty cycle
    pub fn get_effective_duty_cycle(&self) -> f64 {
        match self.scheme {
            ModulationScheme::Continuous => 1.0,
            ModulationScheme::Pulsed | ModulationScheme::DutyCycleControl => self.control.duty_cycle,
            ModulationScheme::BurstMode => {
                // Approximate duty cycle for burst mode
                let burst_duty = 0.02 / 0.1; // burst_duration / burst_period
                burst_duty * self.control.duty_cycle
            }
            _ => self.control.duty_cycle,
        }
    }
}