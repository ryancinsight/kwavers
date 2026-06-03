//! Absorption models for acoustic simulations
//!
//! Numerical models for simulating absorption and dispersion in various media.

use kwavers_core::constants::numerical::CM_TO_M;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::constants::{DB_TO_NP, REFERENCE_FREQUENCY_HZ};
use serde::{Deserialize, Serialize};

/// Absorption models supported by solvers
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum AbsorptionMode {
    /// No absorption
    #[default]
    Lossless,
    /// Stokes absorption (frequency squared)
    Stokes,
    /// Power law absorption: α(f) = α₀ · (f / 1 MHz)^y
    PowerLaw {
        /// Absorption coefficient at 1 MHz [dB/(MHz^y·cm)].
        ///
        /// The spectral solvers convert this raw k-Wave coefficient to the
        /// SI coefficient `Np/((rad/s)^y·m)` at the solver boundary.
        alpha_coeff: f64,
        /// Power law exponent
        alpha_power: f64,
    },
    /// Multi-relaxation absorption model for complex media
    /// References: Szabo, T. L. (1995). "Time domain wave equations for lossy media"
    MultiRelaxation {
        tau: Vec<f64>,     // Relaxation times [s]
        weights: Vec<f64>, // Relaxation weights [dimensionless]
    },
    /// Causal absorption with configurable relaxation times
    /// References: Chen, W. & Holm, S. (2003). "Modified Szabo's wave equation models"
    Causal {
        relaxation_times: Vec<f64>, // Multiple relaxation times [s]
        alpha_0: f64,               // Low-frequency absorption [Np/m]
    },
}

/// Convert the k-Wave power-law absorption coefficient from
/// `dB/(MHz^y·cm)` to the spectral coefficient `Np/((rad/s)^y·m)`.
///
/// # Theorem
/// For a power-law absorption prefactor `α_dB` defined at `1 MHz`,
/// the Treeby & Cox spectral coefficient `α_SI` satisfies:
///
/// ```text
/// α(f) [dB/cm] = α_dB · (f / 1 MHz)^y
/// α(ω) [Np/m]  = α_SI · ω^y
/// α_SI = α_dB · (ln(10)/20) · 100 · (1 / (2π · 10^6))^y
/// ```
///
/// The last line is the exact `k-Wave` `db2neper` conversion used before the
/// fractional-Laplacian coefficients `τ` and `η` are formed.
#[must_use]
#[inline]
pub fn power_law_db_cm_to_np_omega_m(alpha_db_cm: f64, alpha_power: f64) -> f64 {
    // The `/ CM_TO_M` factor converts dB/cm → dB/m (1/0.01 = 100).
    alpha_db_cm * DB_TO_NP / CM_TO_M * (1.0 / (TWO_PI * REFERENCE_FREQUENCY_HZ)).powf(alpha_power)
}

/// Inverse of [`power_law_db_cm_to_np_omega_m`] at a specific frequency: the
/// `dB/(MHz^y·cm)` power-law prefactor that reproduces a given amplitude
/// attenuation `α(f)` [Np/m] at frequency `f` for power-law exponent `y`.
///
/// From `α(f) [Np/m] = α_db · (ln10/20)·100 · (f/1 MHz)^y`:
/// ```text
///   α_db = α_Np_m · (CM_TO_M / DB_TO_NP) · (f_ref / f)^y
/// ```
/// Used to inject a non-power-law excess attenuation (e.g. a resonant
/// bubble-cloud α from Commander–Prosperetti) into a power-law medium's
/// prefactor so it produces that attenuation at the drive frequency. Exact at
/// `f`; off-frequency it then scales as the medium's `f^y`.
#[must_use]
#[inline]
pub fn np_m_to_power_law_db_cm(alpha_np_m: f64, freq_hz: f64, alpha_power: f64) -> f64 {
    if !freq_hz.is_finite() || freq_hz <= 0.0 {
        return 0.0;
    }
    alpha_np_m * (CM_TO_M / DB_TO_NP) * (REFERENCE_FREQUENCY_HZ / freq_hz).powf(alpha_power)
}

#[cfg(test)]
mod tests {
    use super::{np_m_to_power_law_db_cm, power_law_db_cm_to_np_omega_m};

    #[test]
    fn test_power_law_db_cm_to_np_omega_m_matches_kwave_reference() {
        let got = power_law_db_cm_to_np_omega_m(0.75, 1.5);
        let expected = 5.482_481_235_081_536e-10;
        assert!(
            (got - expected).abs() < 1e-24,
            "conversion mismatch: got {got}, expected {expected}"
        );
    }

    #[test]
    fn np_m_to_db_cm_round_trips_at_frequency() {
        // A target attenuation [Np/m] at a drive frequency must convert to a
        // power-law prefactor that reproduces exactly that attenuation at f.
        let alpha_target_np_m = 37.0;
        let f = 0.5e6;
        let y = 1.1;
        let db_cm = np_m_to_power_law_db_cm(alpha_target_np_m, f, y);
        let alpha_si = power_law_db_cm_to_np_omega_m(db_cm, y);
        let omega = 2.0 * std::f64::consts::PI * f;
        let recovered = alpha_si * omega.powf(y);
        assert!(
            (recovered - alpha_target_np_m).abs() <= 1e-9 * alpha_target_np_m,
            "round-trip: target={alpha_target_np_m} Np/m, recovered={recovered} Np/m"
        );
    }

    #[test]
    fn np_m_to_db_cm_guards_nonpositive_frequency() {
        assert_eq!(np_m_to_power_law_db_cm(10.0, 0.0, 1.1), 0.0);
        assert_eq!(np_m_to_power_law_db_cm(10.0, -1.0, 1.1), 0.0);
    }
}
