//! Bubble oscillation response data structures and analysis

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
        if self.scattered_pressure.is_empty() {
            return 0.0;
        }

        // FFT-based harmonic analysis for nonlinear scattering
        let n = self.scattered_pressure.len();
        let fundamental_freq = sample_rate / n as f64;
        let harmonic_freq = fundamental_freq * harmonic as f64;

        // Compute RMS power at harmonic frequency
        let mut real_sum = 0.0;
        let mut imag_sum = 0.0;

        for (i, &pressure) in self.scattered_pressure.iter().enumerate() {
            let phase = 2.0 * std::f64::consts::PI * harmonic_freq * self.time[i];
            real_sum += pressure * phase.cos();
            imag_sum += pressure * phase.sin();
        }

        (real_sum * real_sum + imag_sum * imag_sum).sqrt() / n as f64
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
