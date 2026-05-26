use super::constants::{AIR_POLYTROPIC_INDEX, CAVITATION_PROBABILITY_STEEPNESS};
use super::TherapyCavitationDetector;
use crate::core::constants::fundamental::{ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL};
use crate::core::constants::numerical::{TWO_PI};

impl TherapyCavitationDetector {
    /// Minnaert resonance frequency for a bubble of radius `r0` (m) (Hz).
    ///
    /// ## Theorem (Minnaert 1933)
    /// ```text
    ///   f₀ = (1 / 2πR₀) √(3γP₀ / ρ_L)
    /// ```
    /// For R₀ = 1 µm, γ = 1.4, P₀ = 101 325 Pa, ρ_L = 1000 kg/m³: f₀ ≈ 3.26 MHz.
    #[must_use]
    pub fn minnaert_frequency(&self, r0: f64) -> f64 {
        let pressure_term =
            3.0 * AIR_POLYTROPIC_INDEX * ATMOSPHERIC_PRESSURE / DENSITY_WATER_NOMINAL;
        pressure_term.sqrt() / (TWO_PI * r0)
    }

    /// Cavitation index CI = |P_neg| / P_Blake.
    ///
    /// CI < 0.5: no cavitation; 0.5 ≤ CI < 1.0: stable cavitation; CI ≥ 1.0: inertial cavitation.
    #[must_use]
    pub fn cavitation_index(&self, peak_negative_pressure: f64) -> f64 {
        peak_negative_pressure.abs() / self.blake_threshold
    }

    /// Logistic cavitation probability: sigmoid centred at CI = 1.
    ///
    /// P(cav) = 1 / (1 + exp(−5·(CI − 1)))
    #[must_use]
    pub fn cavitation_probability(&self, peak_negative_pressure: f64) -> f64 {
        let ci = self.cavitation_index(peak_negative_pressure);
        1.0 / (1.0 + (-CAVITATION_PROBABILITY_STEEPNESS * (ci - 1.0)).exp())
    }

    /// Returns `true` when 0.5 < CI < 1.0 (stable/non-inertial cavitation).
    #[must_use]
    pub fn is_stable_cavitation(&self, peak_negative_pressure: f64) -> bool {
        let ci = self.cavitation_index(peak_negative_pressure);
        ci > 0.5 && ci < 1.0
    }

    /// Returns `true` when CI ≥ 1.0 (inertial / transient cavitation).
    #[must_use]
    pub fn is_inertial_cavitation(&self, peak_negative_pressure: f64) -> bool {
        self.cavitation_index(peak_negative_pressure) >= 1.0
    }
}
