//! Quantum correction magnitudes for SBSL emission models.

use super::constants::{C, E_CHARGE, KB, LAMB_SHIFT_HYDROGEN_EV, M_E};
use super::einstein::EinsteinCoefficients;

/// Compute the relativistic parameter `k_B T / (m_e c^2)` for plasma electrons.
///
/// Values much smaller than `0.01` imply nonrelativistic bremsstrahlung
/// corrections below the percent scale.
#[must_use]
pub fn relativistic_parameter(temperature_k: f64) -> f64 {
    KB * temperature_k / (M_E * C * C)
}

/// Compute the leading hydrogenic Lamb-shift scale [eV].
///
/// # Formula
///
/// The leading hydrogen-like scaling is `Delta E_Lamb(Z) = Delta E_H Z^4`.
/// This is used only as an order estimate for whether SBSL thermal broadening
/// dominates radiative QED corrections.
#[must_use]
pub fn lamb_shift_ev(z: f64) -> f64 {
    LAMB_SHIFT_HYDROGEN_EV * z.powi(4)
}

/// Magnitude assessment for quantum corrections relative to classical emission.
#[derive(Debug, Clone)]
pub struct QuantumCorrectionAssessment {
    /// Relativistic parameter `k_B T / (m_e c^2)`.
    pub relativistic_parameter: f64,
    /// Lamb shift divided by thermal energy `k_B T`.
    pub lamb_shift_ratio: f64,
    /// Flash duration divided by radiative lifetime.
    pub flash_to_lifetime_ratio: f64,
    /// First-order classical bremsstrahlung accuracy estimate [%].
    pub classical_accuracy_pct: f64,
}

impl QuantumCorrectionAssessment {
    /// Assess quantum corrections for temperature, flash duration, and a transition.
    #[must_use]
    pub fn assess(
        temperature_k: f64,
        flash_duration_s: f64,
        transition_omega: f64,
        oscillator_strength: f64,
    ) -> Self {
        let rel_param = relativistic_parameter(temperature_k);
        let k_t_ev = KB * temperature_k / E_CHARGE;
        let lamb_ratio = lamb_shift_ev(1.0) / k_t_ev;
        let einstein = EinsteinCoefficients::from_oscillator_strength(
            transition_omega,
            oscillator_strength,
            1.0,
            1.0,
        );
        let flash_ratio = flash_duration_s * einstein.a21;
        let classical_accuracy = (1.0 - rel_param).max(0.0) * 100.0;

        Self {
            relativistic_parameter: rel_param,
            lamb_shift_ratio: lamb_ratio,
            flash_to_lifetime_ratio: flash_ratio,
            classical_accuracy_pct: classical_accuracy,
        }
    }

    /// True if classical bremsstrahlung is accurate to within 0.1%.
    #[must_use]
    pub fn classical_bremsstrahlung_adequate(&self) -> bool {
        self.relativistic_parameter < 1e-3
    }

    /// True if Lamb shift is below 1% of thermal energy.
    #[must_use]
    pub fn lamb_shift_negligible(&self) -> bool {
        self.lamb_shift_ratio < 0.01
    }
}
