//! Pulse train signal implementation

use crate::domain::signal::Signal;
use std::f64::consts::PI;

use super::DEFAULT_DUTY_CYCLE;

/// Pulse train - periodic sequence of pulses
#[derive(Debug, Clone)]
pub struct PulseTrain {
    pulse_frequency: f64,
    carrier_frequency: f64,
    duty_cycle: f64,
    amplitude: f64,
    phase: f64,
    pulse_shape: PulseShape,
}

#[derive(Debug, Clone, Copy)]
pub enum PulseShape {
    Rectangular,
    Gaussian { q_factor: f64 },
    Sinc,
}

impl PulseTrain {
    #[must_use]
    pub fn new(pulse_frequency: f64, carrier_frequency: f64, amplitude: f64) -> Self {
        assert!(pulse_frequency > 0.0, "Pulse frequency must be positive");
        assert!(
            carrier_frequency > 0.0,
            "Carrier frequency must be positive"
        );
        assert!(amplitude >= 0.0, "Amplitude must be non-negative");

        Self {
            pulse_frequency,
            carrier_frequency,
            duty_cycle: DEFAULT_DUTY_CYCLE,
            amplitude,
            phase: 0.0,
            pulse_shape: PulseShape::Rectangular,
        }
    }

    #[must_use]
    pub fn with_duty_cycle(mut self, duty_cycle: f64) -> Self {
        assert!(
            duty_cycle > 0.0 && duty_cycle <= 1.0,
            "Duty cycle must be in (0, 1]"
        );
        self.duty_cycle = duty_cycle;
        self
    }

    #[must_use]
    pub fn with_pulse_shape(mut self, shape: PulseShape) -> Self {
        self.pulse_shape = shape;
        self
    }

    fn envelope(&self, t: f64) -> f64 {
        let period = 1.0 / self.pulse_frequency;
        let phase_in_period = (t % period) / period;

        match self.pulse_shape {
            PulseShape::Rectangular => {
                if phase_in_period < self.duty_cycle {
                    1.0
                } else {
                    0.0
                }
            }

            PulseShape::Gaussian { q_factor } => {
                let center = self.duty_cycle / 2.0;
                let width = self.duty_cycle / (2.0 * q_factor);
                let arg = (phase_in_period - center) / width;
                if phase_in_period < self.duty_cycle {
                    (-arg * arg).exp()
                } else {
                    0.0
                }
            }

            PulseShape::Sinc => {
                if phase_in_period < self.duty_cycle {
                    let x = PI * (phase_in_period / self.duty_cycle - 0.5) * 4.0;
                    if x.abs() < 1e-10 {
                        1.0
                    } else {
                        x.sin() / x
                    }
                } else {
                    0.0
                }
            }
        }
    }
}

impl Signal for PulseTrain {
    fn amplitude(&self, t: f64) -> f64 {
        let envelope = self.envelope(t);
        let carrier = (2.0 * PI * self.carrier_frequency * t + self.phase).sin();
        self.amplitude * envelope * carrier
    }

    fn frequency(&self, _t: f64) -> f64 {
        self.carrier_frequency
    }

    fn phase(&self, _t: f64) -> f64 {
        self.phase
    }

    fn duration(&self) -> Option<f64> {
        None // Pulse train is periodic, no fixed duration
    }

    fn clone_box(&self) -> Box<dyn Signal> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pulse_train() {
        let train = PulseTrain::new(1000.0, 1e6, 1.0);

        // Test periodicity
        let t1 = 0.0;
        let t2 = 1e-3; // One period later
        let amp1 = train.amplitude(t1);
        let amp2 = train.amplitude(t2);

        // Values at same phase should be similar (accounting for carrier phase)
        assert!((amp1.abs() - amp2.abs()).abs() < 0.1);
    }
}
