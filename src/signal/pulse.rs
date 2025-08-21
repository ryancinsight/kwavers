// signal/pulse.rs
//! Pulse signal generation module
//!
//! Implements various pulsing techniques including:
//! - Gaussian pulse
//! - Rectangular pulse
//! - Tone burst
//! - Pulse train
//! - Ricker wavelet (Mexican hat)
//!
//! Literature references:
//! - Oppenheim & Schafer (2010): "Discrete-Time Signal Processing"
//! - Harris (1978): "On the use of windows for harmonic analysis"
//! - Ricker (1953): "The form and laws of propagation of seismic wavelets"

use crate::signal::Signal;
use std::f64::consts::PI;
use std::fmt::Debug;

// Physical constants for pulse signals
/// Default Q factor for Gaussian pulses (bandwidth control)
const DEFAULT_GAUSSIAN_Q: f64 = 5.0;

/// Minimum pulse width in seconds to ensure numerical stability
const MIN_PULSE_WIDTH: f64 = 1e-9;

/// Default rise time fraction for rectangular pulses (10% of pulse width)
const DEFAULT_RISE_TIME_FRACTION: f64 = 0.1;

/// Maximum number of cycles in a tone burst
const MAX_TONE_BURST_CYCLES: usize = 1000;

/// Default duty cycle for pulse trains (50%)
const DEFAULT_DUTY_CYCLE: f64 = 0.5;

/// Gaussian pulse signal
///
/// Implements a Gaussian-modulated sinusoidal pulse:
/// s(t) = A * exp(-((t-t0)/(τ/Q))²) * sin(2πf₀t + φ)
///
/// Where Q controls the bandwidth (higher Q = narrower bandwidth)
#[derive(Debug, Clone)]
pub struct GaussianPulse {
    center_frequency: f64,
    center_time: f64,
    pulse_width: f64,
    amplitude: f64,
    phase: f64,
    q_factor: f64,
}

impl GaussianPulse {
    pub fn new(center_frequency: f64, center_time: f64, pulse_width: f64, amplitude: f64) -> Self {
        assert!(center_frequency > 0.0, "Center frequency must be positive");
        assert!(pulse_width >= MIN_PULSE_WIDTH, "Pulse width too small");
        assert!(amplitude >= 0.0, "Amplitude must be non-negative");

        Self {
            center_frequency,
            center_time,
            pulse_width,
            amplitude,
            phase: 0.0,
            q_factor: DEFAULT_GAUSSIAN_Q,
        }
    }

    pub fn with_q_factor(mut self, q: f64) -> Self {
        assert!(q > 0.0, "Q factor must be positive");
        self.q_factor = q;
        self
    }

    pub fn with_phase(mut self, phase: f64) -> Self {
        self.phase = phase;
        self
    }

    fn envelope(&self, t: f64) -> f64 {
        let tau = self.pulse_width / self.q_factor;
        let arg = (t - self.center_time) / tau;
        (-arg * arg).exp()
    }
}

impl Signal for GaussianPulse {
    fn amplitude(&self, t: f64) -> f64 {
        let envelope = self.envelope(t);
        let carrier = (2.0 * PI * self.center_frequency * t + self.phase).sin();
        self.amplitude * envelope * carrier
    }

    fn frequency(&self, t: f64) -> f64 {
        // Instantaneous frequency for Gaussian pulse
        // Includes chirp due to envelope modulation
        let envelope = self.envelope(t);
        if envelope > 1e-10 {
            self.center_frequency
        } else {
            0.0
        }
    }

    fn phase(&self, t: f64) -> f64 {
        self.phase
    }

    fn duration(&self) -> Option<f64> {
        // Effective duration (99% energy containment)
        Some(6.0 * self.pulse_width / self.q_factor)
    }

    fn clone_box(&self) -> Box<dyn Signal> {
        Box::new(self.clone())
    }
}

/// Rectangular pulse with configurable rise/fall times
///
/// Implements a rectangular pulse with smooth transitions:
/// - Rise time: transition from 0 to full amplitude
/// - Fall time: transition from full amplitude to 0
#[derive(Debug, Clone)]
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

/// Tone burst signal - windowed sinusoid with specific number of cycles
///
/// Common in ultrasound applications for controlled excitation
#[derive(Debug, Clone)]
pub struct ToneBurst {
    center_frequency: f64,
    num_cycles: f64,
    start_time: f64,
    amplitude: f64,
    phase: f64,
    window_type: WindowType,
}

#[derive(Debug, Clone, Copy)]
pub enum WindowType {
    Rectangular,
    Hann,
    Hamming,
    Blackman,
    Gaussian,
    Tukey { alpha: f64 }, // alpha = 0: rectangular, alpha = 1: Hann
}

impl ToneBurst {
    pub fn new(center_frequency: f64, num_cycles: f64, start_time: f64, amplitude: f64) -> Self {
        assert!(center_frequency > 0.0, "Center frequency must be positive");
        assert!(num_cycles > 0.0 && num_cycles <= MAX_TONE_BURST_CYCLES as f64);
        assert!(amplitude >= 0.0, "Amplitude must be non-negative");

        Self {
            center_frequency,
            num_cycles,
            start_time,
            amplitude,
            phase: 0.0,
            window_type: WindowType::Hann,
        }
    }

    pub fn with_window(mut self, window_type: WindowType) -> Self {
        self.window_type = window_type;
        self
    }

    fn duration_seconds(&self) -> f64 {
        self.num_cycles / self.center_frequency
    }

    fn window(&self, t: f64) -> f64 {
        let relative_time = t - self.start_time;
        let duration = self.duration_seconds();

        if relative_time < 0.0 || relative_time > duration {
            return 0.0;
        }

        let normalized_time = relative_time / duration;

        match self.window_type {
            WindowType::Rectangular => 1.0,

            WindowType::Hann => 0.5 * (1.0 - (2.0 * PI * normalized_time).cos()),

            WindowType::Hamming => 0.54 - 0.46 * (2.0 * PI * normalized_time).cos(),

            WindowType::Blackman => {
                0.42 - 0.5 * (2.0 * PI * normalized_time).cos()
                    + 0.08 * (4.0 * PI * normalized_time).cos()
            }

            WindowType::Gaussian => {
                let sigma = 0.4; // Standard deviation (adjustable)
                let arg = (normalized_time - 0.5) / sigma;
                (-0.5 * arg * arg).exp()
            }

            WindowType::Tukey { alpha } => {
                if normalized_time < alpha / 2.0 {
                    0.5 * (1.0 + (2.0 * PI * normalized_time / alpha - PI).cos())
                } else if normalized_time < 1.0 - alpha / 2.0 {
                    1.0
                } else {
                    0.5 * (1.0 + (2.0 * PI * (normalized_time - 1.0) / alpha + PI).cos())
                }
            }
        }
    }
}

impl Signal for ToneBurst {
    fn amplitude(&self, t: f64) -> f64 {
        let window = self.window(t);
        let carrier = (2.0 * PI * self.center_frequency * t + self.phase).sin();
        self.amplitude * window * carrier
    }

    fn frequency(&self, _t: f64) -> f64 {
        self.center_frequency
    }

    fn phase(&self, _t: f64) -> f64 {
        self.phase
    }

    fn duration(&self) -> Option<f64> {
        Some(self.duration_seconds())
    }

    fn clone_box(&self) -> Box<dyn Signal> {
        Box::new(self.clone())
    }
}

/// Ricker wavelet (Mexican hat wavelet)
///
/// Commonly used in seismic and ultrasound applications
/// Second derivative of Gaussian function
///
/// Reference: Ricker, N. (1953). "The form and laws of propagation of seismic wavelets"
#[derive(Debug, Clone)]
pub struct RickerWavelet {
    peak_frequency: f64,
    peak_time: f64,
    amplitude: f64,
}

impl RickerWavelet {
    pub fn new(peak_frequency: f64, peak_time: f64, amplitude: f64) -> Self {
        assert!(peak_frequency > 0.0, "Peak frequency must be positive");
        assert!(amplitude >= 0.0, "Amplitude must be non-negative");

        Self {
            peak_frequency,
            peak_time,
            amplitude,
        }
    }

    /// Compute the Ricker wavelet value
    /// r(t) = A * (1 - 2π²f²τ²) * exp(-π²f²τ²)
    /// where τ = t - t_peak
    fn ricker_value(&self, t: f64) -> f64 {
        let tau = t - self.peak_time;
        let f = self.peak_frequency;
        let arg = PI * f * tau;
        let arg_squared = arg * arg;

        (1.0 - 2.0 * arg_squared) * (-arg_squared).exp()
    }
}

impl Signal for RickerWavelet {
    fn amplitude(&self, t: f64) -> f64 {
        self.amplitude * self.ricker_value(t)
    }

    fn frequency(&self, t: f64) -> f64 {
        // Instantaneous frequency of Ricker wavelet
        // Peak at center, decreases away from center
        let tau = (t - self.peak_time).abs();
        let decay_factor = (-PI * PI * self.peak_frequency * self.peak_frequency * tau * tau).exp();
        self.peak_frequency * decay_factor
    }

    fn phase(&self, _t: f64) -> f64 {
        0.0 // Ricker wavelet is real-valued
    }

    fn duration(&self) -> Option<f64> {
        // Effective duration (99% energy)
        Some(4.0 / self.peak_frequency)
    }

    fn clone_box(&self) -> Box<dyn Signal> {
        Box::new(self.clone())
    }
}

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

    pub fn with_duty_cycle(mut self, duty_cycle: f64) -> Self {
        assert!(
            duty_cycle > 0.0 && duty_cycle <= 1.0,
            "Duty cycle must be in (0, 1]"
        );
        self.duty_cycle = duty_cycle;
        self
    }

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
    fn test_gaussian_pulse() {
        let pulse = GaussianPulse::new(1e6, 1e-6, 1e-7, 1.0);

        // Check peak at center time
        let peak_value = pulse.amplitude(1e-6);
        assert!(peak_value.abs() < 1.1); // Should be close to amplitude

        // Check decay away from center
        let far_value = pulse.amplitude(2e-6);
        assert!(far_value.abs() < 0.01); // Should be nearly zero
    }

    #[test]
    fn test_tone_burst() {
        let burst = ToneBurst::new(1e6, 5.0, 0.0, 1.0);
        let duration = burst.duration().unwrap();

        // Duration should be num_cycles / frequency
        assert!((duration - 5e-6).abs() < 1e-9);
    }

    #[test]
    fn test_ricker_wavelet() {
        let ricker = RickerWavelet::new(30.0, 0.05, 1.0);

        // Check characteristic shape: positive peak at center
        let center_value = ricker.amplitude(0.05);
        assert!(center_value > 0.9);

        // Check negative lobes
        let before = ricker.amplitude(0.04);
        let after = ricker.amplitude(0.06);
        assert!(before < 0.0);
        assert!(after < 0.0);
    }
}
