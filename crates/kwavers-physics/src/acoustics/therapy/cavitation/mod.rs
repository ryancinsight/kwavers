//! Cavitation detection and monitoring for therapeutic ultrasound
//!
//! ## Physical Foundation
//!
//! ### Blake Threshold
//!
//! The critical negative pressure for cavitation inception in a liquid with a
//! spherical nucleus of equilibrium radius R₀ is (Apfel 1981, eq. 7):
//! ```text
//!   P_Blake = |P₀ + P_v − 2σ/R₀|
//! ```
//! where P₀ = ambient pressure, P_v = vapour pressure, σ = surface tension.
//!
//! ### Minnaert Resonance Frequency
//!
//! The natural frequency of a spherical gas bubble is (Minnaert 1933):
//! ```text
//!   f₀ = (1 / 2πR₀) √(3γP₀ / ρ_L)
//! ```
//!
//! ### Spectral Threshold
//!
//! Near Minnaert resonance, bubble oscillations are amplified by the mechanical
//! Q-factor of the oscillator (Leighton 1994, §4.4). The effective threshold is
//! `P_eff = P_Blake / E(f)`. Away from resonance `E → 1` and detection falls
//! back to the Blake threshold.
//!
//! ## References
//! - Minnaert M (1933). Phil. Mag. **16**:235–248. (resonance frequency)
//! - Apfel RE (1981). J. Acoust. Soc. Am. **69**(6):1624–1633. (Blake threshold)
//! - Flynn HG (1982). J. Acoust. Soc. Am. **72**(6):1926–1932. (inertial cavitation)
//! - Leighton TG (1994). *The Acoustic Bubble*. Academic Press. §4.4 (Q-factor)

mod constants;
mod detection;
mod metrics;
mod types;

#[cfg(test)]
mod tests;

pub use types::{CavitationDetectionMethod, TherapyCavitationDetector};

use aequitas::systems::si::quantities::{Frequency, Length, Pressure};

impl TherapyCavitationDetector {
    /// Create a cavitation detector for `frequency` using a 1 µm reference nucleus.
    #[must_use]
    pub fn new(frequency: Frequency<f64>) -> Self {
        Self::new_with_radius(
            frequency,
            Length::from_base(constants::DEFAULT_NUCLEUS_RADIUS),
        )
    }

    /// Create a cavitation detector for `frequency` and nucleus radius `r0`.
    ///
    /// ## Theorem (Blake threshold, Apfel 1981 eq. 7)
    ///
    /// The critical negative pressure for cavitation inception is:
    /// ```text
    ///   P_Blake = |P₀ + P_v − 2σ/R₀|
    /// ```
    /// For R₀ = 1 µm: P_Blake ≈ 41 935 Pa.
    #[must_use]
    pub fn new_with_radius(frequency: Frequency<f64>, r0: Length<f64>) -> Self {
        let blake_threshold = Pressure::from_base(constants::blake_threshold(r0.into_base()));
        Self {
            frequency,
            blake_threshold,
            method: CavitationDetectionMethod::PressureThreshold,
        }
    }
}
