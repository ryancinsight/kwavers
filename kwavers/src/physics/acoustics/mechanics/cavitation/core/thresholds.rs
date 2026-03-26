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
    FLYNN_COLLAPSE_COEFFICIENT * (ambient_pressure + 2.0 * surface_tension / nucleus_radius)
        - vapor_pressure
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
