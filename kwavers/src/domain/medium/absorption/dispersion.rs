//! Dispersion correction models

use ndarray::{Array3, Zip};
use num_complex::Complex;

/// Dispersion model for frequency-dependent phase velocity
#[derive(Debug, Clone)]
pub struct DispersionModel {
    /// Reference sound speed [m/s]
    c0: f64,
    /// Absorption coefficient at reference frequency
    alpha_0: f64,
    /// Power law exponent
    y: f64,
}

impl DispersionModel {
    /// Create a new dispersion model
    #[must_use]
    pub fn new(c0: f64, alpha_0: f64, y: f64) -> Self {
        Self { c0, alpha_0, y }
    }

    /// Calculate phase velocity using Kramers-Kronig relations
    #[must_use]
    pub fn phase_velocity(&self, frequency: f64) -> f64 {
        // For power law absorption, the phase velocity is:
        // c(ω) = c₀ / [1 - (α₀/ω₀^y) * tan(πy/2) * ω^y]

        if frequency == 0.0 {
            return self.c0;
        }

        let omega = 2.0 * std::f64::consts::PI * frequency;
        let tan_factor = (std::f64::consts::PI * self.y / 2.0).tan();

        // Avoid numerical issues for y close to 2
        if (self.y - 2.0).abs() < 1e-10 {
            self.c0
        } else {
            self.c0 / (1.0 - self.alpha_0 * tan_factor * omega.powf(self.y - 1.0))
        }
    }

    /// Calculate group velocity
    #[must_use]
    pub fn group_velocity(&self, frequency: f64) -> f64 {
        // Group velocity: vg = c²/vp + ω * d(vp)/dω
        let vp = self.phase_velocity(frequency);

        // For small frequency changes, use numerical derivative
        let df = frequency * 1e-6; // Small perturbation
        let vp_plus = self.phase_velocity(frequency + df);
        let dvp_df = (vp_plus - vp) / df;

        let omega = 2.0 * std::f64::consts::PI * frequency;
        vp + omega * dvp_df
    }
}

/// Dispersion correction in k-space
#[derive(Debug)]
pub struct DispersionCorrection {
    model: DispersionModel,
}

impl DispersionCorrection {
    /// Create a new dispersion correction
    #[must_use]
    pub fn new(model: DispersionModel) -> Self {
        Self { model }
    }

    /// Apply dispersion correction in k-space
    pub fn apply_k_space(
        &self,
        spectrum: &mut Array3<Complex<f64>>,
        k_values: &Array3<f64>,
        dt: f64,
    ) {
        Zip::from(spectrum).and(k_values).for_each(|s, &k| {
            if k != 0.0 {
                // Frequency corresponding to this k value
                let freq = k * self.model.c0 / (2.0 * std::f64::consts::PI);

                // Phase velocity at this frequency
                let c_phase = self.model.phase_velocity(freq.abs());

                // Dispersion phase shift
                let phase_shift = k * (c_phase - self.model.c0) * dt;

                // Apply phase correction
                *s *= Complex::from_polar(1.0, phase_shift);
            }
        });
    }

    /// Calculate dispersion relation k(ω)
    #[must_use]
    pub fn dispersion_relation(&self, frequency: f64) -> f64 {
        let c_phase = self.model.phase_velocity(frequency);
        2.0 * std::f64::consts::PI * frequency / c_phase
    }
}
