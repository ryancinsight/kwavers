// frequency_sweep/chirp.rs - Linear frequency sweep (chirp)

use super::{constants::*, FrequencySweep, SweepDirection};
use crate::signal::Signal;
use std::f64::consts::PI;

/// Linear frequency sweep (chirp signal)
///
/// Frequency varies linearly with time:
/// f(t) = f₀ + (f₁ - f₀) * t / T
///
/// Phase: φ(t) = 2π * [f₀*t + (f₁-f₀)*t²/(2T)]
#[derive(Debug, Clone)]
pub struct LinearChirp {
    start_frequency: f64,
    stop_frequency: f64,
    duration: f64,
    amplitude: f64,
    sweep_rate: f64,
    direction: SweepDirection,
}

impl LinearChirp {
    /// Create new linear chirp - USING all parameters
    pub fn new(start_freq: f64, stop_freq: f64, duration: f64, amplitude: f64) -> Self {
        assert!(start_freq > 0.0, "Start frequency must be positive");
        assert!(stop_freq > 0.0, "Stop frequency must be positive");
        assert!(duration > MIN_SWEEP_DURATION, "Duration too short");
        assert!(amplitude >= 0.0, "Amplitude must be non-negative");

        let sweep_rate = (stop_freq - start_freq) / duration;
        let direction = if stop_freq > start_freq {
            SweepDirection::Upward
        } else {
            SweepDirection::Downward
        };

        Self {
            start_frequency: start_freq,
            stop_frequency: stop_freq,
            duration,
            amplitude,
            sweep_rate,
            direction,
        }
    }

    /// Get sweep direction
    pub fn direction(&self) -> SweepDirection {
        self.direction
    }
}

impl Signal for LinearChirp {
    fn amplitude(&self, t: f64) -> f64 {
        if t < 0.0 || t > self.duration {
            return 0.0;
        }

        let phase = self.phase(t);
        self.amplitude * phase.sin()
    }

    fn duration(&self) -> Option<f64> {
        Some(self.duration)
    }
}

impl FrequencySweep for LinearChirp {
    fn instantaneous_frequency(&self, t: f64) -> f64 {
        if t < 0.0 || t > self.duration {
            return 0.0;
        }
        self.start_frequency + self.sweep_rate * t
    }

    fn phase(&self, t: f64) -> f64 {
        if t < 0.0 || t > self.duration {
            return 0.0;
        }
        TWO_PI * (self.start_frequency * t + 0.5 * self.sweep_rate * t * t)
    }

    fn sweep_rate(&self, t: f64) -> f64 {
        if t < 0.0 || t > self.duration {
            0.0
        } else {
            self.sweep_rate
        }
    }

    fn start_frequency(&self) -> f64 {
        self.start_frequency
    }

    fn stop_frequency(&self) -> f64 {
        self.stop_frequency
    }

    fn duration(&self) -> f64 {
        self.duration
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_chirp() {
        let chirp = LinearChirp::new(1000.0, 2000.0, 0.001, 1.0);

        // Check frequencies
        assert!((chirp.instantaneous_frequency(0.0) - 1000.0).abs() < FREQUENCY_TOLERANCE);
        assert!((chirp.instantaneous_frequency(0.001) - 2000.0).abs() < FREQUENCY_TOLERANCE);

        // Check sweep rate
        assert!((chirp.sweep_rate(0.0005) - 1e6).abs() < 1.0);
    }
}
