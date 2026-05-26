//! Bubble oscillation response data structures and analysis

use crate::core::constants::numerical::{TWO_PI};
/// Bubble oscillation response data
#[derive(Debug, Clone)]
pub struct BubbleResponse {
    /// Time points (s)
    pub time: Vec<f64>,
    /// Bubble radius over time (m)
    pub radius: Vec<f64>,
    /// Scattered acoustic pressure (Pa)
    pub scattered_pressure: Vec<f64>,
}

impl BubbleResponse {
    /// Compute harmonic content of scattered signal
    #[must_use]
    pub fn harmonic_content(&self, harmonic: usize, sample_rate: f64) -> f64 {
        if harmonic == 0
            || self.scattered_pressure.is_empty()
            || self.time.len() != self.scattered_pressure.len()
            || !positive_finite(sample_rate)
            || !self.time.iter().all(|value| value.is_finite())
            || !self
                .scattered_pressure
                .iter()
                .all(|value| value.is_finite())
        {
            return 0.0;
        }

        // FFT-based harmonic analysis for nonlinear scattering
        let n = self.scattered_pressure.len();
        let fundamental_freq = sample_rate / n as f64;
        let harmonic_freq = fundamental_freq * harmonic as f64;

        // Compute RMS power at harmonic frequency
        let mut real_sum = 0.0;
        let mut imag_sum = 0.0;

        for (&time, &pressure) in self.time.iter().zip(self.scattered_pressure.iter()) {
            let phase = TWO_PI * harmonic_freq * time;
            real_sum += pressure * phase.cos();
            imag_sum += pressure * phase.sin();
        }

        let content = real_sum.hypot(imag_sum) / n as f64;
        if content.is_finite() {
            content
        } else {
            0.0
        }
    }

    /// Get maximum radius excursion
    #[must_use]
    pub fn max_radius_change(&self) -> f64 {
        if self.radius.is_empty() {
            return 0.0;
        }

        let eq_radius = self.radius[0]; // Assume starts at equilibrium
        self.radius
            .iter()
            .map(|&r| (r - eq_radius).abs())
            .fold(0.0, f64::max)
    }
}

#[inline]
fn positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

#[cfg(test)]
mod tests {
    use super::BubbleResponse;

    #[test]
    fn harmonic_content_recovers_unit_sine_dft_coefficient() {
        let samples = 64usize;
        let sample_rate = 64.0;
        let harmonic = 2usize;
        let time = (0..samples)
            .map(|index| index as f64 / sample_rate)
            .collect::<Vec<_>>();
        let scattered_pressure = time
            .iter()
            .map(|time| (2.0 * std::f64::consts::PI * harmonic as f64 * time).sin())
            .collect::<Vec<_>>();
        let response = BubbleResponse {
            time,
            radius: Vec::new(),
            scattered_pressure,
        };

        let content = response.harmonic_content(harmonic, sample_rate);

        assert!((content - 0.5).abs() < 1e-12);
    }

    #[test]
    fn harmonic_content_rejects_invalid_domains_with_zero() {
        let valid = BubbleResponse {
            time: vec![0.0, 0.25, 0.5, 0.75],
            radius: Vec::new(),
            scattered_pressure: vec![0.0, 1.0, 0.0, -1.0],
        };

        assert_eq!(valid.harmonic_content(0, 4.0), 0.0);
        assert_eq!(valid.harmonic_content(1, 0.0), 0.0);
        assert_eq!(valid.harmonic_content(1, f64::NAN), 0.0);

        let mismatched = BubbleResponse {
            time: vec![0.0],
            radius: Vec::new(),
            scattered_pressure: vec![0.0, 1.0],
        };
        assert_eq!(mismatched.harmonic_content(1, 4.0), 0.0);

        let nonfinite_time = BubbleResponse {
            time: vec![0.0, f64::INFINITY],
            radius: Vec::new(),
            scattered_pressure: vec![0.0, 1.0],
        };
        assert_eq!(nonfinite_time.harmonic_content(1, 4.0), 0.0);

        let nonfinite_pressure = BubbleResponse {
            time: vec![0.0, 0.25],
            radius: Vec::new(),
            scattered_pressure: vec![0.0, f64::NAN],
        };
        assert_eq!(nonfinite_pressure.harmonic_content(1, 4.0), 0.0);
    }
}
