//! Cavitation threshold models
//!
//! # Mathematical Specifications
//!
//! ## Theorem 1: Blake Threshold (Static Stability)
//! A gas bubble of undeformed radius $R_0$ in a liquid subjected to decreasing
//! ambient pressure becomes mechanically unstable and grows explosively when
//! the local static pressure drops below the Blake critical pressure $P_B$.
//!
//! **Derivation:**
//! The initial gas pressure satisfies the Young-Laplace condition:
//! $P_{g0} = P_0 - P_v + 2\sigma/R_0$
//! Instability occurs at the saddle point of the pressure-radius curve, giving
//! the critical radius $R_c = R_0\sqrt{3P_{g0}R_0/(2\sigma)}$ and
//! $$ P_B = P_v - \frac{4\sigma}{3R_0}\sqrt{\frac{2\sigma}{3P_{g0}R_0}} $$
//!
//! The **acoustic amplitude threshold** returned by [`blake_threshold`] is:
//! $$ P_\text{Blake} = P_0 - P_B $$
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

/// Calculate the Blake threshold: the acoustic rarefaction amplitude [Pa] at
/// which a gas nucleus of radius `initial_radius` undergoes unbounded growth.
///
/// # Derivation (Blake 1949; Apfel 1984)
///
/// For a spherical nucleus in mechanical equilibrium with the surrounding liquid,
/// the total pressure that must be overcome to cause unbounded growth is found by
/// locating the saddle-point of the Rayleigh-Plesset pressure curve:
///
/// ```text
/// P_g0 = P_0 − P_v + 2σ/R_0                 [Young-Laplace equilibrium]
/// R_c  = R_0 · √(3 P_g0 R_0 / (2σ))          [critical radius]
/// P_B  = P_v − (4σ/(3R_0)) · √(2σ/(3 P_g0 R_0))   [Blake static pressure]
/// ```
///
/// The function returns the **acoustic amplitude threshold** (positive, Pa):
///
/// ```text
/// P_Blake = P_0 − P_B = (P_0 − P_v) + (4σ/(3R_0)) · √(2σ/(3 P_g0 R_0))
/// ```
///
/// Cavitation is initiated when the local acoustic pressure `p` satisfies
/// `p < −P_Blake` (rarefaction exceeds the threshold).
///
/// # Physical limits
/// - R_0 → ∞: P_Blake → P_0 − P_v  (surface tension negligible)
/// - R_0 → 0: P_Blake → ∞  (tiny nuclei are very stable)
#[must_use]
pub fn blake_threshold(
    surface_tension: f64,  // [N/m]
    initial_radius: f64,   // [m]
    ambient_pressure: f64, // [Pa]
    vapor_pressure: f64,   // [Pa]
) -> f64 {
    // Equilibrium gas pressure (Young-Laplace)
    let p_g0 = (ambient_pressure - vapor_pressure + 2.0 * surface_tension / initial_radius)
        .max(f64::EPSILON);

    // Blake critical static pressure:  P_B = P_v − (4σ/(3R_0)) · √(2σ/(3 P_g0 R_0))
    let p_blake = vapor_pressure
        - (4.0 * surface_tension / (3.0 * initial_radius))
            * (2.0 * surface_tension / (3.0 * p_g0 * initial_radius)).sqrt();

    // Return acoustic amplitude threshold (always non-negative)
    (ambient_pressure - p_blake).max(0.0)
}

/// Blake critical radius `R_c` \[m]: the radius at which a gas nucleus of
/// undeformed radius `initial_radius` crosses the static-stability saddle point
/// and begins unbounded growth (Chapter 5 Theorem 5.3). Companion to
/// [`blake_threshold`], which returns the pressure at the same saddle point.
///
/// ```text
/// P_g0 = P_0 − P_v + 2σ/R_0           [Young-Laplace equilibrium gas pressure]
/// R_c  = R_0 · √(3 P_g0 R_0 / (2σ))
/// ```
///
/// `R_c > R_0` whenever `P_0 > P_v` (the bubble must grow past equilibrium to go
/// unstable). Returns `f64::INFINITY` for non-positive surface tension or radius
/// (no restoring curvature ⇒ no finite saddle).
#[must_use]
pub fn blake_critical_radius(
    surface_tension: f64,  // [N/m]
    initial_radius: f64,   // [m]
    ambient_pressure: f64, // [Pa]
    vapor_pressure: f64,   // [Pa]
) -> f64 {
    if surface_tension <= 0.0 || initial_radius <= 0.0 {
        return f64::INFINITY;
    }
    let p_g0 = (ambient_pressure - vapor_pressure + 2.0 * surface_tension / initial_radius)
        .max(f64::EPSILON);
    initial_radius * (3.0 * p_g0 * initial_radius / (2.0 * surface_tension)).sqrt()
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
    use kwavers_core::constants::cavitation::FLYNN_COLLAPSE_COEFFICIENT;
    // P_Flynn = α * (P_0 + 2σ/R_n) - P_v, where α ≈ 0.83 (Flynn 1964)
    FLYNN_COLLAPSE_COEFFICIENT.mul_add(
        ambient_pressure + 2.0 * surface_tension / nucleus_radius,
        -vapor_pressure,
    )
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
    use kwavers_core::constants::cavitation::{
        FLYNN_COLLAPSE_COEFFICIENT, SURFACE_TENSION_WATER, VAPOR_PRESSURE_WATER,
    };
    use kwavers_core::constants::fundamental::ATMOSPHERIC_PRESSURE;

    // Physical constants for a 5 μm air bubble in water at 20°C
    const SIGMA: f64 = SURFACE_TENSION_WATER; // 0.0728 N/m (cavitation::SURFACE_TENSION_WATER)
    const R0: f64 = 5e-6; // m
    const P0: f64 = ATMOSPHERIC_PRESSURE; // Pa (1 atm)
    const PV: f64 = VAPOR_PRESSURE_WATER; // 2339.0 Pa (cavitation::VAPOR_PRESSURE_WATER)

    /// Blake threshold: acoustic amplitude = P_0 − P_B where
    /// P_B = P_v − (4σ/(3R_0)) · √(2σ/(3 P_g0 R_0))
    /// and P_g0 = P_0 − P_v + 2σ/R_0.
    ///
    /// For 5 µm bubble in water at 20 °C (SSOT: P₀=101325 Pa, Pᵥ=2339 Pa, σ=0.0728 N/m):
    /// P_g0 = (101325 − 2339) + 2·0.0728/5e-6 = 98986 + 29120 = 128106 Pa
    /// α = √(2σ/(3·P_g0·R_0)) = √(0.1456/1.9216) ≈ 0.27527
    /// threshold = (P_0 − P_v) + (4σ/(3R_0))·α = 98986 + 19413·0.27527 ≈ 104328 Pa
    #[test]
    fn blake_threshold_matches_analytical_formula() {
        let p_g0 = P0 - PV + 2.0 * SIGMA / R0;
        let p_blake = PV - (4.0 * SIGMA / (3.0 * R0)) * (2.0 * SIGMA / (3.0 * p_g0 * R0)).sqrt();
        let expected = P0 - p_blake;
        let got = blake_threshold(SIGMA, R0, P0, PV);
        // Primary correctness check: output equals the analytical formula evaluated at SSOT constants.
        assert!(
            (got - expected).abs() < 1.0,
            "Blake threshold: got {got:.1} expected {expected:.1}"
        );
        // Numerical sanity: threshold must be positive (stable nuclei need rarefaction to cavitate)
        assert!(got > 0.0, "Blake threshold must be positive (got {got:.1})");
        // Range check: threshold > (P₀ − Pᵥ) because surface tension adds a positive offset.
        assert!(got > P0 - PV - 1.0, "Blake > P_0 − P_v for finite R_0");
    }

    /// Blake critical radius R_c = R_0·√(3 P_g0 R_0/(2σ)) (Theorem 5.3), the
    /// companion saddle-point quantity to the Blake threshold.
    #[test]
    fn blake_critical_radius_matches_theorem() {
        let p_g0 = P0 - PV + 2.0 * SIGMA / R0;
        let expected = R0 * (3.0 * p_g0 * R0 / (2.0 * SIGMA)).sqrt();
        let got = blake_critical_radius(SIGMA, R0, P0, PV);
        assert!(
            (got - expected).abs() < 1e-15,
            "R_c: got {got:e} expected {expected:e}"
        );
        // The critical radius exceeds the equilibrium radius (the bubble must grow
        // past equilibrium to cross the instability saddle).
        assert!(got > R0, "Blake critical radius {got:e} must exceed R_0 {R0:e}");
        // Non-physical surface tension / radius ⇒ no finite saddle.
        assert!(blake_critical_radius(0.0, R0, P0, PV).is_infinite());
        assert!(blake_critical_radius(SIGMA, 0.0, P0, PV).is_infinite());
    }

    /// Neppiras threshold: P_N = 0.5 · ((P₀ − Pᵥ) + 2σ/R₀).
    ///
    /// With SSOT constants (P₀=101325 Pa, Pᵥ=2339 Pa, σ=0.0728 N/m, R₀=5 µm):
    /// 0.5 · (98986 + 29120) = 0.5 · 128106 = 64053.0 Pa.
    #[test]
    fn neppiras_threshold_matches_analytical_formula() {
        let expected = 0.5 * ((P0 - PV) + 2.0 * SIGMA / R0);
        let got = neppiras_threshold(P0, PV, SIGMA, R0);
        // Primary correctness check: output equals the analytical formula evaluated at SSOT constants.
        assert!(
            (got - expected).abs() < 1.0,
            "Neppiras threshold: got {got:.1} expected {expected:.1}"
        );
        // Range check: threshold must lie in (0, P0) for physically reasonable parameters.
        assert!(
            got > 0.0 && got < P0,
            "Neppiras threshold out of range: {got:.1}"
        );
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

    /// `flynn_criterion`: violent collapse when R_max/R₀ > 2.
    #[test]
    fn flynn_criterion_true_above_two_and_false_below() {
        assert!(
            flynn_criterion(2.1 * R0, R0),
            "R_max/R0=2.1 must trigger violent collapse"
        );
        assert!(
            !flynn_criterion(1.9 * R0, R0),
            "R_max/R0=1.9 must not trigger violent collapse"
        );
        assert!(
            !flynn_criterion(2.0 * R0, R0),
            "R_max/R0=2.0 exactly is not > 2 → false"
        );
    }
}
