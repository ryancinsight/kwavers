//! Validity-regime guards for the Bjerknes force laws, and the
//! Duryea et al. aggregation-regime classifier.
//!
//! # Why guards are needed here
//!
//! The secondary Bjerknes force implemented in this module
//! ([`BjerknesCalculator::secondary_bjerknes_force`]) is the
//! time-averaged **linear** form derived from Crum (1975,
//! *J. Acoust. Soc. Am.* 57, 536–538):
//! `⟨F⟩ = −(ρ/(4πd²))·⟨V̇₁V̇₂⟩`, which for harmonic oscillation
//! reduces to `−(ρω²/(8π))·V₁V₂·cos(φ)/d²` with `Vᵢ` the peak volume
//! oscillation amplitudes — the line-of-centres component, positive
//! away from the other bubble, so in-phase pairs attract (`⟨F⟩ < 0`).
//! (The instantaneous Crum form is the canonical kernel
//! [`secondary_bjerknes_force_instantaneous`](super::secondary_bjerknes_force_instantaneous);
//! time-averaging it over a harmonic cycle gives the form here.
//! KW-PHYS-009 item 1, consolidated 2026-11-27.)
//!
//! Linear theory requires **small oscillation amplitude**,
//! `ξ/R0 ≪ 1` (ξ = wall displacement amplitude), and **weak
//! coupling** (a neighbour's radiated pressure ≪ the drive). Neither
//! holds for inertially collapsing cavitation clouds: at the
//! triplet-pulse formation conditions (Build 8 audit,
//! formation/lit_build8_bjerknes.md) `ξ/R0` = 6–209× and the
//! neighbour's radiated pressure is 26–60% of the drive. Strong-field
//! treatments (Mettin et al. 1997; Pelekasis et al. 2004; Lanoy et
//! al. 2015; Pandey 2019; Doinikov 2001) are numerical/qualitative —
//! sign reversal and bound-pair formation — with no closed-form law
//! near 1–1.25 MHz / 6.7–14.7 MPa / 3–6 µm bubbles in tissue. The
//! naive uncoupled-monopole estimate at those conditions predicts a
//! first merge in 0.05 µs, falsified by ≥10⁴× against observed cloud
//! persistence (≥600 µs). [`secondary_bjerknes_force_checked`]
//! therefore refuses pairs whose displacement ratio reaches order
//! unity, where the linear force has no meaning.
//!
//! # Aggregation regime (Duryea et al. 2015)
//!
//! Whether Bjerknes attraction actually consolidates a bubble cloud
//! is a separate, measured question. Duryea et al. (2015, *IEEE
//! Trans. UFFC* 62, 1605–1614) found secondary-Bjerknes aggregation
//! of ~2.8 µm-radius residual bubbles works at 0.2 < MI < 1 (optimum
//! MI 0.8; 92% consolidated at 2 MHz in 0.5 ms) and **fails above
//! MI ≈ 1**: the pulses re-excite the residuals into inertial
//! cavitation, producing more daughters instead of merging them.
//! [`duryea_aggregation_regime`] classifies an operating point
//! against that measured cliff. The formation conditions (MI 6.0
//! Iwasaki / 14.7 Kuji) sit 4–10× beyond the tested range on the
//! failing side, which is why the formation model keeps coalescence
//! OFF during the burst — a modelling decision now grounded in a
//! measured cliff rather than an assumption.

use super::calculator::BjerknesCalculator;
use super::types::BjerknesForceData;
use kwavers_core::error::{KwaversError, KwaversResult, PhysicsError};
use std::f64::consts::PI;

/// Displacement ratio `ξ/R0` at which the linear Bjerknes law is
/// refused by [`secondary_bjerknes_force_checked`]: at order-unity
/// displacement the small-amplitude derivation has no meaning (the
/// formation-regime values are 6–209×).
pub const BJERKNES_LINEAR_MAX_DISPLACEMENT_RATIO: f64 = 1.0;

/// Lower edge of the Duryea et al. (2015) aggregation window
/// (mechanical index).
pub const DURYEA_AGGREGATION_MI_MIN: f64 = 0.2;
/// Upper edge of the Duryea et al. (2015) aggregation window
/// (mechanical index): above MI ≈ 1 aggregation fails.
pub const DURYEA_AGGREGATION_MI_MAX: f64 = 1.0;

/// Aggregation outcome expected at an operating point, per Duryea et
/// al. (2015).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationRegime {
    /// MI < 0.2: below the measured aggregation window (attraction
    /// too weak / too slow to consolidate within the exposure).
    BelowWindow,
    /// 0.2 ≤ MI ≤ 1: aggregation measured to work (optimum MI 0.8).
    AggregationWindow,
    /// MI > 1: above the measured cliff — pulses re-excite residuals
    /// into inertial cavitation and produce more daughters; chained
    /// merging does not occur.
    AboveCliff,
}

/// Classify a mechanical index against the Duryea et al. (2015)
/// aggregation window.
#[must_use]
pub fn duryea_aggregation_regime(mechanical_index: f64) -> AggregationRegime {
    if mechanical_index < DURYEA_AGGREGATION_MI_MIN {
        AggregationRegime::BelowWindow
    } else if mechanical_index <= DURYEA_AGGREGATION_MI_MAX {
        AggregationRegime::AggregationWindow
    } else {
        AggregationRegime::AboveCliff
    }
}

/// Wall displacement amplitude ξ [m] implied by a peak volume
/// oscillation amplitude `ΔV` [m³] at equilibrium radius `radius`:
/// linearised, `ΔV ≈ 4πR0²·ξ`.
#[must_use]
pub fn displacement_amplitude(volume_amplitude: f64, radius: f64) -> f64 {
    if radius <= 0.0 {
        return 0.0;
    }
    volume_amplitude.abs() / (4.0 * PI * radius * radius)
}

/// Linear-regime assessment of a bubble pair for the secondary
/// Bjerknes law.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BjerknesRegime {
    /// `ξ₁/R₁` displacement ratio of bubble 1.
    pub displacement_ratio_1: f64,
    /// `ξ₂/R₂` displacement ratio of bubble 2.
    pub displacement_ratio_2: f64,
}

impl BjerknesRegime {
    /// True while both displacement ratios are below order unity —
    /// the linear law is at least self-consistent (accuracy still
    /// wants `ξ/R0 ≪ 1`; see [`Self::small_amplitude`]).
    #[must_use]
    pub fn linear_regime(&self) -> bool {
        self.displacement_ratio_1 < BJERKNES_LINEAR_MAX_DISPLACEMENT_RATIO
            && self.displacement_ratio_2 < BJERKNES_LINEAR_MAX_DISPLACEMENT_RATIO
    }

    /// True when both displacement ratios are < 0.1, where the
    /// linear law is quantitatively trustworthy.
    #[must_use]
    pub fn small_amplitude(&self) -> bool {
        self.displacement_ratio_1 < 0.1 && self.displacement_ratio_2 < 0.1
    }
}

/// Assess the linear regime of a pair from their radii [m] and peak
/// volume oscillation amplitudes [m³] (the same amplitudes
/// [`BjerknesCalculator::secondary_bjerknes_force`] takes).
#[must_use]
pub fn bjerknes_regime(
    radius_1: f64,
    volume_amplitude_1: f64,
    radius_2: f64,
    volume_amplitude_2: f64,
) -> BjerknesRegime {
    let ratio = |v: f64, r: f64| {
        if r > 0.0 {
            displacement_amplitude(v, r) / r
        } else {
            0.0
        }
    };
    BjerknesRegime {
        displacement_ratio_1: ratio(volume_amplitude_1, radius_1),
        displacement_ratio_2: ratio(volume_amplitude_2, radius_2),
    }
}

/// Radiated pressure amplitude [Pa] of an oscillating bubble at
/// distance `distance` [m]: `p_rad = ρ·ω²·ΔV/(4πd)`. Compare against
/// the drive amplitude for the weak-coupling check (linear pair
/// theory wants `p_rad ≪ P_A`; the formation conditions give
/// 26–60%).
#[must_use]
pub fn radiated_pressure_amplitude(
    volume_amplitude: f64,
    frequency: f64,
    distance: f64,
    density: f64,
) -> f64 {
    if distance <= 0.0 || frequency <= 0.0 {
        return 0.0;
    }
    let omega = 2.0 * PI * frequency;
    density * omega * omega * volume_amplitude.abs() / (4.0 * PI * distance)
}

/// Checked form of
/// [`BjerknesCalculator::secondary_bjerknes_force`]: refuses pairs
/// outside the linear regime instead of returning a force the
/// derivation does not support.
///
/// # Errors
/// - [`KwaversError::Physics`] with `PhysicsError::InvalidParameter`
///   when either bubble's displacement ratio `ξ/R0` reaches
///   [`BJERKNES_LINEAR_MAX_DISPLACEMENT_RATIO`].
/// - Otherwise whatever the underlying calculator returns (its own
///   input validation: non-positive radii/distance).
pub fn secondary_bjerknes_force_checked(
    calculator: &BjerknesCalculator,
    bubble1_radius: f64,
    bubble2_radius: f64,
    volume_amplitude1: f64,
    volume_amplitude2: f64,
    phase_difference: f64,
    distance: f64,
) -> KwaversResult<BjerknesForceData> {
    let regime = bjerknes_regime(
        bubble1_radius,
        volume_amplitude1,
        bubble2_radius,
        volume_amplitude2,
    );
    if !regime.linear_regime() {
        let (parameter, value) =
            if regime.displacement_ratio_1 >= BJERKNES_LINEAR_MAX_DISPLACEMENT_RATIO {
                ("volume_amplitude1 / radius1", regime.displacement_ratio_1)
            } else {
                ("volume_amplitude2 / radius2", regime.displacement_ratio_2)
            };
        return Err(KwaversError::Physics(PhysicsError::InvalidParameter {
            parameter: parameter.to_owned(),
            value,
            reason: format!(
                "Linear secondary Bjerknes law requires displacement ratio xi/R0 < {BJERKNES_LINEAR_MAX_DISPLACEMENT_RATIO}; got {value:.1} (inertial oscillation — no closed-form strong-field law exists)"
            ),
        }));
    }
    calculator.secondary_bjerknes_force(
        bubble1_radius,
        bubble2_radius,
        volume_amplitude1,
        volume_amplitude2,
        phase_difference,
        distance,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acoustics::bubble_dynamics::bjerknes_forces::types::BjerknesConfig;

    #[test]
    fn test_duryea_bands() {
        assert_eq!(
            duryea_aggregation_regime(0.1),
            AggregationRegime::BelowWindow
        );
        assert_eq!(
            duryea_aggregation_regime(0.8),
            AggregationRegime::AggregationWindow
        );
        assert_eq!(
            duryea_aggregation_regime(1.0),
            AggregationRegime::AggregationWindow
        );
        // Formation operating points: far above the cliff.
        assert_eq!(
            duryea_aggregation_regime(6.0),
            AggregationRegime::AboveCliff
        );
        assert_eq!(
            duryea_aggregation_regime(14.7),
            AggregationRegime::AboveCliff
        );
    }

    #[test]
    fn test_displacement_ratio_round_trip() {
        // ΔV built from ξ = 0.05·R0 must return ratio 0.05.
        let r0 = 5e-6;
        let xi = 0.05 * r0;
        let dv = 4.0 * PI * r0 * r0 * xi;
        let regime = bjerknes_regime(r0, dv, r0, dv);
        assert!((regime.displacement_ratio_1 - 0.05).abs() < 1e-12);
        assert!(regime.linear_regime());
        assert!(regime.small_amplitude());
    }

    #[test]
    fn test_inertial_pair_refused() {
        // A bubble expanding to Rmax = 10·R0 has ΔV ≈ V(Rmax):
        // ξ/R0 is in the hundreds — the formation regime.
        let r0: f64 = 5e-6;
        let dv = (4.0 / 3.0) * PI * (10.0 * r0).powi(3);
        let regime = bjerknes_regime(r0, dv, r0, dv);
        assert!(!regime.linear_regime());

        let calc = BjerknesCalculator::new(BjerknesConfig::default());
        // The refusal names the first bubble's ratio and carries its value.
        let refused = secondary_bjerknes_force_checked(&calc, r0, r0, dv, dv, 0.0, 100e-6);
        assert!(matches!(
            refused,
            Err(KwaversError::Physics(PhysicsError::InvalidParameter {
                ref parameter,
                value,
                ..
            })) if parameter == "volume_amplitude1 / radius1"
                && value.to_bits() == regime.displacement_ratio_1.to_bits()
        ));
        // A small-amplitude pair passes the guard: the checked form
        // returns the calculator's own result, field for field.
        let dv_small = 4.0 * PI * r0 * r0 * (0.05 * r0);
        let checked =
            secondary_bjerknes_force_checked(&calc, r0, r0, dv_small, dv_small, 0.0, 50e-6)
                .unwrap();
        let plain = calc
            .secondary_bjerknes_force(r0, r0, dv_small, dv_small, 0.0, 50e-6)
            .unwrap();
        assert_eq!(checked.primary.to_bits(), plain.primary.to_bits());
        assert_eq!(checked.secondary.to_bits(), plain.secondary.to_bits());
        assert_eq!(checked.total.to_bits(), plain.total.to_bits());
        assert_eq!(
            checked.phase_difference.to_bits(),
            plain.phase_difference.to_bits()
        );
        assert_eq!(checked.interaction_type, plain.interaction_type);
        assert_eq!(checked.distance.to_bits(), plain.distance.to_bits());
        assert_eq!(checked.coalescing, plain.coalescing);
    }

    #[test]
    fn test_radiated_pressure_scales() {
        // p_rad ∝ ΔV/d: doubling distance halves it.
        let p1 = radiated_pressure_amplitude(1e-15, 1.0e6, 100e-6, 1000.0);
        let p2 = radiated_pressure_amplitude(1e-15, 1.0e6, 200e-6, 1000.0);
        assert!((p1 - 2.0 * p2).abs() / p1 < 1e-12);
        assert!(p1 > 0.0);
    }
}
