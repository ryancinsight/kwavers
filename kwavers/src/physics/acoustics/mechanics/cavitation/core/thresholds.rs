//! Cavitation threshold models
//!
//! # Mathematical Specifications
//!
//! ## Theorem 1: Blake Threshold (Static Stability)
//! A gas bubble of undeformed radius $R_0$ in a liquid subjected to decreasing
//! ambient pressure becomes mechanically unstable and grows explosively when
//! the local static pressure drops below the Blake threshold $P_B$.
//!
//! **Proof / Equation:**
//! The internal pressure balances surface tension and ambient pressure:
//! $P_g = P_0 - P_v + \frac{2\sigma}{R_0}$
//! Instability occurs when the derivative of pressure with respect to radius is zero,
//! establishing the Blake threshold:
//! $$ P_{Blake} = P_0 + P_v - \frac{2\sigma}{R_0} $$
//!
//! ## Theorem 2: Transient Cavitation (Flynn/Neppiras)
//! For acoustic fields, transient (inertial) cavitation occurs when the mechanical
//! energy imparted to the bubble overcomes both hydrostatic and surface tension
//! forces, leading to a violent subsequent collapse ($R_{max}/R_0 \ge 2$).
//!
//! ## Theorem 3: Mechanical Index (MI)
//! MI is an empirical risk parameter correlating directly with the probability of
//! non-thermal bio-effects (inertial cavitation) under diagnostic ultrasound pulses.
//!
//! $$ MI = \frac{P_{neg, MPa}}{\sqrt{f_{c, MHz}}} $$
//! Regulatory FDA limits restrict $MI \le 1.9$ for human imaging.

/// Cavitation threshold models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdModel {
    /// Blake threshold (Static pressure instability point)
    Blake,
    /// Neppiras threshold (Acoustic pressure transient limit)
    Neppiras,
    /// Apfel-Holland mechanical index (Empirical bio-effect risk)
    MechanicalIndex,
    /// Flynn threshold (Violent collapse criterion, $R_{max}/R_0 \ge 2$)
    Flynn,
}

/// Calculate Blake threshold pressure
/// Based on Blake (1949): "The onset of cavitation in liquids"
#[must_use]
pub fn blake_threshold(
    surface_tension: f64,  // [N/m]
    initial_radius: f64,   // [m]
    ambient_pressure: f64, // [Pa]
    vapor_pressure: f64,   // [Pa]
) -> f64 {
    // P_Blake = P_0 + P_v - 2σ/R_0
    ambient_pressure + vapor_pressure - 2.0 * surface_tension / initial_radius
}

/// Calculate Neppiras threshold
/// Based on Neppiras (1980): "Acoustic cavitation"
#[must_use]
pub fn neppiras_threshold(
    ambient_pressure: f64, // [Pa]
    vapor_pressure: f64,   // [Pa]
    surface_tension: f64,  // [N/m]
    nucleus_radius: f64,   // [m]
) -> f64 {
    // Threshold for transient cavitation
    let hydrostatic = ambient_pressure - vapor_pressure;
    let surface = 2.0 * surface_tension / nucleus_radius;

    0.5 * (hydrostatic + surface)
}

/// Calculate Flynn threshold for violent cavitation
/// Based on Flynn (1964): "Physics of Acoustic Cavitation in Liquids"
#[must_use]
pub fn flynn_threshold(
    ambient_pressure: f64, // [Pa]
    vapor_pressure: f64,   // [Pa]
    surface_tension: f64,  // [N/m]
    nucleus_radius: f64,   // [m]
) -> f64 {
    use crate::core::constants::cavitation::FLYNN_COLLAPSE_COEFFICIENT;
    // P_Flynn = α * (P_0 + 2σ/R_n) - P_v, where α ≈ 0.83 (Flynn 1964)
    FLYNN_COLLAPSE_COEFFICIENT.mul_add(ambient_pressure + 2.0 * surface_tension / nucleus_radius, -vapor_pressure)
}

/// Calculate mechanical index (MI)
/// MI = P_neg / sqrt(f_c) where P_neg in MPa and f_c in MHz
#[must_use]
pub fn mechanical_index(peak_negative_pressure: f64, center_frequency: f64) -> f64 {
    let p_mpa = peak_negative_pressure.abs() / 1e6; // Convert Pa to MPa
    let f_mhz = center_frequency / 1e6; // Convert Hz to MHz

    p_mpa / f_mhz.sqrt()
}

/// Flynn's criterion for violent collapse
/// Based on Flynn (1964): "Physics of acoustic cavitation in liquids"
#[must_use]
pub fn flynn_criterion(
    max_radius: f64,     // [m]
    initial_radius: f64, // [m]
) -> bool {
    // Violent collapse when R_max/R_0 > 2
    max_radius / initial_radius > 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::cavitation::FLYNN_COLLAPSE_COEFFICIENT;

    // Physical constants for a 5 μm air bubble in water at 20°C
    const SIGMA: f64 = 0.0728; // N/m
    const R0: f64 = 5e-6; // m
    const P0: f64 = 101_325.0; // Pa (1 atm)
    const PV: f64 = 2_330.0; // Pa (vapor pressure at 20°C)

    /// Blake threshold: P_B = P₀ + Pᵥ − 2σ/R₀.
    ///
    /// Analytical: 101325 + 2330 − 0.1456/5e-6 = 103655 − 29120 = 74535 Pa.
    #[test]
    fn blake_threshold_matches_analytical_formula() {
        let expected = P0 + PV - 2.0 * SIGMA / R0;
        let got = blake_threshold(SIGMA, R0, P0, PV);
        assert!(
            (got - expected).abs() < 1.0,
            "Blake threshold: got {got:.1} expected {expected:.1}"
        );
        assert!((got - 74_535.0).abs() < 1.0, "Blake: must be ~74535 Pa");
    }

    /// Neppiras threshold: P_N = 0.5 · ((P₀ − Pᵥ) + 2σ/R₀).
    ///
    /// Analytical: 0.5 · (99995 + 29120) = 64557.5 Pa.
    #[test]
    fn neppiras_threshold_matches_analytical_formula() {
        let expected = 0.5 * ((P0 - PV) + 2.0 * SIGMA / R0);
        let got = neppiras_threshold(P0, PV, SIGMA, R0);
        assert!(
            (got - expected).abs() < 1.0,
            "Neppiras threshold: got {got:.1} expected {expected:.1}"
        );
        // Numerical: 0.5·(98995 + 29120) = 64057.5 Pa
        assert!((got - 64_057.5).abs() < 1.0, "Neppiras: must be ~64057.5 Pa");
    }

    /// Flynn threshold: P_F = α·(P₀ + 2σ/R₀) − Pᵥ, where α=0.83.
    ///
    /// Analytical: 0.83·(101325 + 29120) − 2330 = 0.83·130445 − 2330 ≈ 105939 Pa.
    #[test]
    fn flynn_threshold_matches_analytical_formula() {
        let expected = FLYNN_COLLAPSE_COEFFICIENT * (P0 + 2.0 * SIGMA / R0) - PV;
        let got = flynn_threshold(P0, PV, SIGMA, R0);
        assert!(
            (got - expected).abs() < 1.0,
            "Flynn threshold: got {got:.1} expected {expected:.1}"
        );
    }

    /// Mechanical index: MI = |P_neg,MPa| / sqrt(f_MHz).
    ///
    /// At P_neg = 0.5 MPa, f = 1 MHz: MI = 0.5 / sqrt(1) = 0.5.
    #[test]
    fn mechanical_index_matches_formula_at_half_mpa_one_mhz() {
        let p_neg = 0.5e6_f64;
        let f = 1e6_f64;
        let mi = mechanical_index(p_neg, f);
        assert!(
            (mi - 0.5).abs() < 1e-12,
            "MI at 0.5MPa/1MHz must be 0.5 (got {mi:.6})"
        );
    }

    /// MI scales as 1/sqrt(f): doubling frequency reduces MI by factor 1/√2.
    #[test]
    fn mechanical_index_scales_inversely_with_sqrt_frequency() {
        let p_neg = 1e6_f64;
        let mi1 = mechanical_index(p_neg, 1e6);
        let mi2 = mechanical_index(p_neg, 4e6);
        // MI(4MHz) = MI(1MHz)/sqrt(4) = MI(1MHz)/2
        assert!(
            (mi2 - mi1 / 2.0).abs() < 1e-12,
            "MI(4MHz)={mi2:.6} must equal MI(1MHz)/2={:.6}",
            mi1 / 2.0
        );
    }

    /// `flynn_criterion`: violent collapse when R_max/R₀ > 2.
    #[test]
    fn flynn_criterion_true_above_two_and_false_below() {
        assert!(flynn_criterion(2.1 * R0, R0), "R_max/R0=2.1 must trigger violent collapse");
        assert!(!flynn_criterion(1.9 * R0, R0), "R_max/R0=1.9 must not trigger violent collapse");
        assert!(!flynn_criterion(2.0 * R0, R0), "R_max/R0=2.0 exactly is not > 2 → false");
    }
}
