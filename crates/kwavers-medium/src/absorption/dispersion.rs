//! Dispersion correction models

use kwavers_core::constants::numerical::TWO_PI;
use leto::Array3;
use eunomia::Complex;

use crate::parallel::zip_mut_ref;

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

    /// Phase velocity from the Szabo / Treeby-Cox Kramers-Kronig relation
    /// for power-law absorption.
    ///
    /// ## Formula (Szabo 1994 Eq 13; Treeby & Cox 2010 Eq 11)
    ///
    /// For an absorption law `α(ω) = α₀·|ω|^y` (with `α₀` in Np/(m·(rad/s)^y)),
    /// causality + Kramers-Kronig give
    ///
    /// ```text
    /// c_p(ω)⁻¹ − c_0⁻¹ = α₀ · tan(πy/2) · (|ω|^(y-1) − |ω_ref|^(y-1))
    /// ```
    ///
    /// Inverting and taking the low-reference limit `ω_ref → 0`:
    ///
    /// ```text
    /// c_p(ω) = c_0 / (1 + c_0 · α₀ · tan(πy/2) · |ω|^(y-1))
    /// ```
    ///
    /// The sign convention follows Szabo (1994): `tan(πy/2) < 0` for
    /// `y ∈ (1, 2)` (the soft-tissue regime), so `c_p > c_0` — anomalous
    /// dispersion, matching measurements.
    ///
    /// Prior to 2026-05-21 this used `(1 − α₀·tan·ω^(y-1))` (minus sign)
    /// and omitted the `c_0` factor; the inconsistent sign drove dispersion
    /// in the *opposite* direction relative to Szabo, and the missing `c_0`
    /// scaling collapsed the correction to ~0.01 % of its physical magnitude
    /// for typical tissue parameters (vs the expected ~16 % at 1 MHz).
    ///
    /// ## References
    /// - Szabo T. L. (1994). *J. Acoust. Soc. Am.* 96(1), 491–500. Eq 13.
    /// - Treeby B. E. & Cox B. T. (2010). *J. Acoust. Soc. Am.* 127(5), 2741. Eq 11.
    #[must_use]
    pub fn phase_velocity(&self, frequency: f64) -> f64 {
        if frequency == 0.0 {
            return self.c0;
        }

        let omega = TWO_PI * frequency;
        let tan_factor = (std::f64::consts::PI * self.y / 2.0).tan();

        // y → 2 makes tan(πy/2) → ±∞ (vertical asymptote); no finite dispersion
        // correction is defined exactly at y = 2.  Caller-side reasoning: for
        // pure-quadratic (water-like) absorption, dispersion is exactly zero.
        if (self.y - 2.0).abs() < 1e-10 {
            return self.c0;
        }

        let denom = (self.c0 * self.alpha_0 * tan_factor).mul_add(omega.powf(self.y - 1.0), 1.0);
        self.c0 / denom
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

        let omega = TWO_PI * frequency;
        vp + omega * dvp_df
    }
}

/// Dispersion correction in k-space
#[derive(Debug)]
pub struct AbsorptionDispersionCorrection {
    model: DispersionModel,
}

impl AbsorptionDispersionCorrection {
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
        zip_mut_ref(spectrum, k_values, |s, &k| {
            if k != 0.0 {
                // Frequency corresponding to this k value
                let freq = k * self.model.c0 / (TWO_PI);

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
        TWO_PI * frequency / c_phase
    }
}

