//! Einstein transition coefficients for two-level atomic systems.

use std::f64::consts::PI;

use super::constants::{C, EPS0, E_CHARGE, HBAR, M_E};
use crate::core::constants::numerical::TWO_PI;

/// Einstein A and B coefficients for a two-level atomic transition.
///
/// # Definition
///
/// For transition angular frequency `omega21 = (E2 - E1) / hbar`:
///
/// ```text
/// A21 = omega21^3 |d12|^2 / (3 pi eps0 hbar c^3)
/// B12 = pi |d12|^2 / (3 eps0 hbar^2)
/// B21 = (g1 / g2) B12
/// ```
///
/// The oscillator-strength form is
/// `|d12|^2 = 3 hbar e^2 f12 / (2 m_e omega21)` and
/// `A21 = (g1/g2) e^2 omega21^2 f12 / (2 pi eps0 m_e c^3)`.
#[derive(Debug, Clone)]
pub struct EinsteinCoefficients {
    /// Spontaneous emission rate A21 [s^-1].
    pub a21: f64,
    /// Stimulated emission coefficient B21 [m^3 J^-1 s^-2].
    pub b21: f64,
    /// Absorption coefficient B12 [m^3 J^-1 s^-2].
    pub b12: f64,
    /// Transition angular frequency omega21 [rad/s].
    pub omega21: f64,
    /// Transition dipole moment squared |d12|^2 [C^2 m^2].
    pub dipole_moment_sq: f64,
}

impl EinsteinCoefficients {
    /// Compute Einstein coefficients from transition frequency and oscillator strength.
    ///
    /// # Theorem: detailed balance degeneracy relation
    ///
    /// In thermodynamic equilibrium, upward and downward stimulated transition
    /// rates satisfy `g1 B12 = g2 B21`. Therefore `B12 / B21 = g2 / g1`.
    /// The spontaneous rate carries the inverse ratio because `f12` is defined
    /// over the lower-level substate ensemble.
    ///
    /// # Domain
    ///
    /// Valid inputs require positive finite `omega21`, `f12`, `g1`, and `g2`.
    /// Invalid inputs produce non-finite coefficients instead of silently
    /// substituting arbitrary denominator values.
    #[must_use]
    pub fn from_oscillator_strength(omega21: f64, f12: f64, g1: f64, g2: f64) -> Self {
        if !(omega21.is_finite()
            && f12.is_finite()
            && g1.is_finite()
            && g2.is_finite()
            && omega21 > 0.0
            && f12 > 0.0
            && g1 > 0.0
            && g2 > 0.0)
        {
            return Self {
                a21: f64::NAN,
                b21: f64::NAN,
                b12: f64::NAN,
                omega21,
                dipole_moment_sq: f64::NAN,
            };
        }

        let dipole_sq = 3.0 * HBAR * E_CHARGE * E_CHARGE * f12 / (2.0 * M_E * omega21);
        let a21 = (E_CHARGE * E_CHARGE * omega21 * omega21 * f12 * g1)
            / (TWO_PI * EPS0 * M_E * C * C * C * g2);
        // B12 (absorption) = π|d12|² / (3 ε₀ ℏ²)   [Rybicki & Lightman 1979, §1.6]
        // B21 (stimulated emission) = (g1/g2) B12    [detailed balance: g1 B12 = g2 B21]
        let b12 = (PI * dipole_sq) / (3.0 * EPS0 * HBAR * HBAR);
        let b21 = b12 * g1 / g2;

        Self {
            a21,
            b21,
            b12,
            omega21,
            dipole_moment_sq: dipole_sq,
        }
    }

    /// Radiative lifetime `tau = 1 / A21` (s).
    #[must_use]
    pub fn radiative_lifetime(&self) -> f64 {
        if self.a21 <= 0.0 || !self.a21.is_finite() {
            f64::INFINITY
        } else {
            1.0 / self.a21
        }
    }

    /// Fraction of excited atoms that emit spontaneously during `dt` (s).
    ///
    /// The Poisson survival law gives `f_emit = 1 - exp(-A21 dt)`.
    #[must_use]
    pub fn flash_emission_fraction(&self, dt: f64) -> f64 {
        1.0 - (-self.a21 * dt).exp()
    }
}
