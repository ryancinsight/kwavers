// signal/modulation.rs
//! Signal modulation techniques module
//!
//! Implements various modulation schemes:
//! - Amplitude Modulation (AM)
//! - Frequency Modulation (FM)
//! - Phase Modulation (PM)
//! - Quadrature Amplitude Modulation (QAM)
//! - Pulse Width Modulation (PWM)
//!
//! Literature references:
//! - Proakis & Salehi (2008): "Digital Communications"
//! - Carlson et al. (2002): "Communication Systems"
//! - Haykin (2001): "Communication Systems"

use crate::signal::Signal;
use std::f64::consts::PI;
use std::fmt::Debug;

// Modulation constants
/// Default modulation index for AM
const DEFAULT_AM_MODULATION_INDEX: f64 = 0.5;

/// Default modulation index for FM (in radians)
const DEFAULT_FM_MODULATION_INDEX: f64 = 5.0;

/// Default modulation index for PM (in radians)
const DEFAULT_PM_MODULATION_INDEX: f64 = 1.0;

/// Maximum modulation index to prevent overmodulation
const MAX_MODULATION_INDEX: f64 = 10.0;

/// Minimum carrier frequency to ensure proper modulation
const MIN_CARRIER_FREQUENCY: f64 = 100.0;

/// Amplitude Modulation (AM)
///
/// Modulates the amplitude of a carrier signal with a modulating signal:
/// s(t) = A_c[1 + m·m(t)]·cos(2πf_c·t + φ)
///
/// Where:
/// - A_c: carrier amplitude
/// - m: modulation index (0 < m ≤ 1 for standard AM)
/// - m(t): modulating signal (normalized to [-1, 1])
/// - f_c: carrier frequency
#[derive(Debug, Clone)]
pub struct AmplitudeModulation {
    carrier_frequency: f64,
    carrier_amplitude: f64,
    carrier_phase: f64,
    modulation_index: f64,
    modulating_signal: Box<dyn Signal>,
    suppressed_carrier: bool,
}

impl AmplitudeModulation {
    pub fn new(
        carrier_frequency: f64,
        carrier_amplitude: f64,
        modulating_signal: Box<dyn Signal>,
    ) -> Self {
        assert!(
            carrier_frequency >= MIN_CARRIER_FREQUENCY,
            "Carrier frequency too low"
        );
        assert!(
            carrier_amplitude > 0.0,
            "Carrier amplitude must be positive"
        );

        Self {
            carrier_frequency,
            carrier_amplitude,
            carrier_phase: 0.0,
            modulation_index: DEFAULT_AM_MODULATION_INDEX,
            modulating_signal,
            suppressed_carrier: false,
        }
    }

    pub fn with_modulation_index(mut self, index: f64) -> Self {
        assert!(
            index > 0.0 && index <= MAX_MODULATION_INDEX,
            "Invalid modulation index"
        );
        self.modulation_index = index;
        self
    }

    pub fn with_suppressed_carrier(mut self, suppressed: bool) -> Self {
        self.suppressed_carrier = suppressed;
        self
    }

    pub fn with_carrier_phase(mut self, phase: f64) -> Self {
        self.carrier_phase = phase;
        self
    }
}

impl Signal for AmplitudeModulation {
    fn amplitude(&self, t: f64) -> f64 {
        let modulating = self.modulating_signal.amplitude(t);
        let carrier = (2.0 * PI * self.carrier_frequency * t + self.carrier_phase).cos();

        if self.suppressed_carrier {
            // Double-sideband suppressed carrier (DSB-SC)
            self.carrier_amplitude * self.modulation_index * modulating * carrier
        } else {
            // Standard AM with carrier
            self.carrier_amplitude * (1.0 + self.modulation_index * modulating) * carrier
        }
    }

    fn frequency(&self, _t: f64) -> f64 {
        self.carrier_frequency
    }

    fn phase(&self, _t: f64) -> f64 {
        self.carrier_phase
    }

    fn duration(&self) -> Option<f64> {
        self.modulating_signal.duration()
    }

    fn clone_box(&self) -> Box<dyn Signal> {
        Box::new(self.clone())
    }
}

/// Frequency Modulation (FM)
///
/// Modulates the frequency of a carrier signal:
/// s(t) = A_c·cos(2πf_c·t + β∫m(τ)dτ + φ)
///
/// Where:
/// - β: modulation index (frequency deviation / modulating frequency)
/// - Instantaneous frequency: f_i(t) = f_c + β·f_m·m(t)
#[derive(Debug, Clone)]
pub struct FrequencyModulation {
    carrier_frequency: f64,
    carrier_amplitude: f64,
    carrier_phase: f64,
    frequency_deviation: f64,
    modulating_signal: Box<dyn Signal>,
    integration_samples: usize,
}

impl FrequencyModulation {
    pub fn new(
        carrier_frequency: f64,
        carrier_amplitude: f64,
        frequency_deviation: f64,
        modulating_signal: Box<dyn Signal>,
    ) -> Self {
        assert!(
            carrier_frequency >= MIN_CARRIER_FREQUENCY,
            "Carrier frequency too low"
        );
        assert!(
            carrier_amplitude > 0.0,
            "Carrier amplitude must be positive"
        );
        assert!(
            frequency_deviation > 0.0,
            "Frequency deviation must be positive"
        );

        Self {
            carrier_frequency,
            carrier_amplitude,
            carrier_phase: 0.0,
            frequency_deviation,
            modulating_signal,
            integration_samples: 1000, // Default integration resolution
        }
    }

    pub fn with_carrier_phase(mut self, phase: f64) -> Self {
        self.carrier_phase = phase;
        self
    }

    pub fn with_integration_samples(mut self, samples: usize) -> Self {
        assert!(samples > 0, "Integration samples must be positive");
        self.integration_samples = samples;
        self
    }

    /// Calculate the phase modulation due to frequency modulation
    fn integrated_phase_modulation(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }

        // Numerical integration using trapezoidal rule
        let dt = t / self.integration_samples as f64;
        let mut integral = 0.0;

        for i in 0..self.integration_samples {
            let t1 = i as f64 * dt;
            let t2 = (i + 1) as f64 * dt;
            let m1 = self.modulating_signal.amplitude(t1);
            let m2 = self.modulating_signal.amplitude(t2);
            integral += 0.5 * (m1 + m2) * dt;
        }

        2.0 * PI * self.frequency_deviation * integral
    }
}

impl Signal for FrequencyModulation {
    fn amplitude(&self, t: f64) -> f64 {
        let phase_modulation = self.integrated_phase_modulation(t);
        let total_phase =
            2.0 * PI * self.carrier_frequency * t + phase_modulation + self.carrier_phase;
        self.carrier_amplitude * total_phase.cos()
    }

    fn frequency(&self, t: f64) -> f64 {
        // Instantaneous frequency
        let modulating = self.modulating_signal.amplitude(t);
        self.carrier_frequency + self.frequency_deviation * modulating
    }

    fn phase(&self, t: f64) -> f64 {
        self.integrated_phase_modulation(t) + self.carrier_phase
    }

    fn duration(&self) -> Option<f64> {
        self.modulating_signal.duration()
    }

    fn clone_box(&self) -> Box<dyn Signal> {
        Box::new(self.clone())
    }
}

/// Phase Modulation (PM)
///
/// Modulates the phase of a carrier signal:
/// s(t) = A_c·cos(2πf_c·t + β·m(t) + φ)
///
/// Where β is the phase deviation in radians
#[derive(Debug, Clone)]
pub struct PhaseModulation {
    carrier_frequency: f64,
    carrier_amplitude: f64,
    carrier_phase: f64,
    phase_deviation: f64,
    modulating_signal: Box<dyn Signal>,
}

impl PhaseModulation {
    pub fn new(
        carrier_frequency: f64,
        carrier_amplitude: f64,
        phase_deviation: f64,
        modulating_signal: Box<dyn Signal>,
    ) -> Self {
        assert!(
            carrier_frequency >= MIN_CARRIER_FREQUENCY,
            "Carrier frequency too low"
        );
        assert!(
            carrier_amplitude > 0.0,
            "Carrier amplitude must be positive"
        );
        assert!(phase_deviation > 0.0, "Phase deviation must be positive");

        Self {
            carrier_frequency,
            carrier_amplitude,
            carrier_phase: 0.0,
            phase_deviation,
            modulating_signal,
        }
    }

    pub fn with_carrier_phase(mut self, phase: f64) -> Self {
        self.carrier_phase = phase;
        self
    }
}

impl Signal for PhaseModulation {
    fn amplitude(&self, t: f64) -> f64 {
        let modulating = self.modulating_signal.amplitude(t);
        let phase_modulation = self.phase_deviation * modulating;
        let total_phase =
            2.0 * PI * self.carrier_frequency * t + phase_modulation + self.carrier_phase;
        self.carrier_amplitude * total_phase.cos()
    }

    fn frequency(&self, t: f64) -> f64 {
        // Instantaneous frequency for PM
        // f_i(t) = f_c + (β/2π) * dm(t)/dt
        // Approximate derivative numerically
        let dt = 1e-6;
        let m1 = self.modulating_signal.amplitude(t - dt);
        let m2 = self.modulating_signal.amplitude(t + dt);
        let derivative = (m2 - m1) / (2.0 * dt);

        self.carrier_frequency + self.phase_deviation * derivative / (2.0 * PI)
    }

    fn phase(&self, t: f64) -> f64 {
        let modulating = self.modulating_signal.amplitude(t);
        self.phase_deviation * modulating + self.carrier_phase
    }

    fn duration(&self) -> Option<f64> {
        self.modulating_signal.duration()
    }

    fn clone_box(&self) -> Box<dyn Signal> {
        Box::new(self.clone())
    }
}

/// Quadrature Amplitude Modulation (QAM)
///
/// Combines two amplitude-modulated signals in quadrature:
/// s(t) = I(t)·cos(2πf_c·t) - Q(t)·sin(2πf_c·t)
///
/// Used for transmitting two independent signals on the same carrier
#[derive(Debug, Clone)]
pub struct QuadratureAmplitudeModulation {
    carrier_frequency: f64,
    carrier_amplitude: f64,
    in_phase_signal: Box<dyn Signal>,
    quadrature_signal: Box<dyn Signal>,
}

impl QuadratureAmplitudeModulation {
    pub fn new(
        carrier_frequency: f64,
        carrier_amplitude: f64,
        in_phase_signal: Box<dyn Signal>,
        quadrature_signal: Box<dyn Signal>,
    ) -> Self {
        assert!(
            carrier_frequency >= MIN_CARRIER_FREQUENCY,
            "Carrier frequency too low"
        );
        assert!(
            carrier_amplitude > 0.0,
            "Carrier amplitude must be positive"
        );

        Self {
            carrier_frequency,
            carrier_amplitude,
            in_phase_signal,
            quadrature_signal,
        }
    }
}

impl Signal for QuadratureAmplitudeModulation {
    fn amplitude(&self, t: f64) -> f64 {
        let i_component = self.in_phase_signal.amplitude(t);
        let q_component = self.quadrature_signal.amplitude(t);

        let carrier_cos = (2.0 * PI * self.carrier_frequency * t).cos();
        let carrier_sin = (2.0 * PI * self.carrier_frequency * t).sin();

        self.carrier_amplitude * (i_component * carrier_cos - q_component * carrier_sin)
    }

    fn frequency(&self, _t: f64) -> f64 {
        self.carrier_frequency
    }

    fn phase(&self, t: f64) -> f64 {
        // Phase of QAM signal
        let i_component = self.in_phase_signal.amplitude(t);
        let q_component = self.quadrature_signal.amplitude(t);

        q_component.atan2(i_component)
    }

    fn duration(&self) -> Option<f64> {
        // Duration is the maximum of both signals
        match (
            self.in_phase_signal.duration(),
            self.quadrature_signal.duration(),
        ) {
            (Some(d1), Some(d2)) => Some(d1.max(d2)),
            (Some(d), None) | (None, Some(d)) => Some(d),
            (None, None) => None,
        }
    }

    fn clone_box(&self) -> Box<dyn Signal> {
        Box::new(self.clone())
    }
}

/// Pulse Width Modulation (PWM)
///
/// Modulates the width of pulses in a pulse train
/// Duty cycle varies with modulating signal
#[derive(Debug, Clone)]
pub struct PulseWidthModulation {
    carrier_frequency: f64,
    amplitude: f64,
    min_duty_cycle: f64,
    max_duty_cycle: f64,
    modulating_signal: Box<dyn Signal>,
}

impl PulseWidthModulation {
    pub fn new(carrier_frequency: f64, amplitude: f64, modulating_signal: Box<dyn Signal>) -> Self {
        assert!(
            carrier_frequency > 0.0,
            "Carrier frequency must be positive"
        );
        assert!(amplitude >= 0.0, "Amplitude must be non-negative");

        Self {
            carrier_frequency,
            amplitude,
            min_duty_cycle: 0.1,
            max_duty_cycle: 0.9,
            modulating_signal,
        }
    }

    pub fn with_duty_cycle_range(mut self, min: f64, max: f64) -> Self {
        assert!(min > 0.0 && min < 1.0, "Min duty cycle must be in (0, 1)");
        assert!(
            max > min && max <= 1.0,
            "Max duty cycle must be in (min, 1]"
        );
        self.min_duty_cycle = min;
        self.max_duty_cycle = max;
        self
    }

    fn get_duty_cycle(&self, t: f64) -> f64 {
        let modulating = self.modulating_signal.amplitude(t);
        // Normalize modulating signal to [0, 1] assuming it's in [-1, 1]
        let normalized = (modulating + 1.0) / 2.0;
        self.min_duty_cycle + (self.max_duty_cycle - self.min_duty_cycle) * normalized
    }
}

impl Signal for PulseWidthModulation {
    fn amplitude(&self, t: f64) -> f64 {
        let period = 1.0 / self.carrier_frequency;
        let phase_in_period = (t % period) / period;
        let duty_cycle = self.get_duty_cycle(t);

        if phase_in_period < duty_cycle {
            self.amplitude
        } else {
            0.0
        }
    }

    fn frequency(&self, _t: f64) -> f64 {
        self.carrier_frequency
    }

    fn phase(&self, _t: f64) -> f64 {
        0.0 // PWM doesn't have a meaningful phase
    }

    fn duration(&self) -> Option<f64> {
        self.modulating_signal.duration()
    }

    fn clone_box(&self) -> Box<dyn Signal> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::sine_wave::SineWave;

    #[test]
    fn test_amplitude_modulation() {
        let modulating = Box::new(SineWave::new(100.0, 1.0, 0.0));
        let am = AmplitudeModulation::new(1000.0, 1.0, modulating).with_modulation_index(0.5);

        // Check that amplitude varies between 0.5 and 1.5
        let samples: Vec<f64> = (0..100).map(|i| am.amplitude(i as f64 * 0.0001)).collect();

        // Envelope should vary
        assert!(samples.iter().any(|&s| s.abs() > 1.2));
        assert!(samples.iter().any(|&s| s.abs() < 0.8));
    }

    #[test]
    fn test_frequency_modulation() {
        let modulating = Box::new(SineWave::new(100.0, 1.0, 0.0));
        let fm = FrequencyModulation::new(1000.0, 1.0, 50.0, modulating);

        // Check instantaneous frequency varies around carrier
        let freq_at_peak = fm.frequency(0.0025); // Peak of 100 Hz sine at t=1/400
        let freq_at_trough = fm.frequency(0.0075); // Trough at t=3/400

        assert!(freq_at_peak > 1000.0);
        assert!(freq_at_trough < 1000.0);
    }

    #[test]
    fn test_phase_modulation() {
        let modulating = Box::new(SineWave::new(100.0, 1.0, 0.0));
        let pm = PhaseModulation::new(1000.0, 1.0, 0.5, modulating);

        // Check phase deviation
        let phase_at_peak = pm.phase(0.0025);
        let phase_at_zero = pm.phase(0.0);

        assert!((phase_at_peak - 0.5).abs() < 0.1); // Should be close to max deviation
        assert!(phase_at_zero.abs() < 0.1); // Should be close to zero
    }
}
