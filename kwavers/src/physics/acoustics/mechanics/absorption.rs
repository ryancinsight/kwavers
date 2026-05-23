//! Absorption models for acoustic simulations
//!
//! Numerical models for simulating absorption and dispersion in various media.

use crate::core::constants::numerical::CM_TO_M;
use crate::core::constants::{DB_TO_NP, REFERENCE_FREQUENCY_HZ};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

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
    alpha_db_cm * DB_TO_NP / CM_TO_M
        * (1.0 / (2.0 * PI * REFERENCE_FREQUENCY_HZ)).powf(alpha_power)
}

#[cfg(test)]
mod tests {
    use super::power_law_db_cm_to_np_omega_m;

    #[test]
    fn test_power_law_db_cm_to_np_omega_m_matches_kwave_reference() {
        let got = power_law_db_cm_to_np_omega_m(0.75, 1.5);
        let expected = 5.482_481_235_081_536e-10;
        assert!(
            (got - expected).abs() < 1e-24,
            "conversion mismatch: got {got}, expected {expected}"
        );
    }
}
