//! Secondary Bjerknes force calculations
//!
//! ## Canonical sign convention (KW-PHYS-009 item 1, consolidated 2026-11-27)
//!
//! The scalar secondary Bjerknes force in this crate is the component,
//! along the line of centres, of the force exerted on one bubble by the
//! other, **positive in the direction pointing away from the other
//! bubble**. With `V̇ᵢ` the volume oscillation rate of bubble `i`,
//! `d` the centre-to-centre distance and `ρ₀` the liquid density, the
//! law (Crum 1975) is
//!
//! ```text
//! F = −(ρ₀/(4πd²)) · ⟨V̇₁V̇₂⟩
//! ```
//!
//! so a pair oscillating **in phase** (`⟨V̇₁V̇₂⟩ > 0`) has `F < 0`:
//! the force on each bubble is directed toward the other —
//! **attraction**. A pair in **antiphase** has `F > 0` — **repulsion**.
//! This is the standard result (Bjerknes 1906; Crum 1975; Leighton
//! 1994 §3.4): Lanoy et al. (2015), Eq. (3) after Crum, state the
//! vector form `F⃗₁→₂ ∝ (r⃗₁ − r⃗₂)·ξ₁ξ₂cos(φ₁ − φ₂)`, whose
//! direction vector points from bubble 2 toward bubble 1 for
//! in-phase pairs — the same statement.
//!
//! Before the consolidation, this module's calculator returned the
//! attraction-positive strength (the negative of the Crum component),
//! while `interactions::BjerknesForceComputer::secondary` returned
//! the Crum component — identical magnitudes, opposite signs. Both
//! now evaluate the single kernel in this file
//! ([`secondary_bjerknes_force_instantaneous`] /
//! [`secondary_bjerknes_force_mean`]) under the convention above.

use super::calculator::BjerknesCalculator;
use super::types::{BjerknesForceData, BjerknesInteractionType};
use kwavers_core::constants::numerical::{FOUR_PI, TWO_PI};
use kwavers_core::error::{KwaversError, KwaversResult};

/// The one canonical secondary Bjerknes kernel (Crum 1975):
/// `F = −ρ₀ · (V̇₁V̇₂) / (4πd²)`, the line-of-centres component of the
/// force on one bubble due to the other, positive away from the other
/// bubble (see the module docs for the convention). `volume_rate_product`
/// is the instantaneous product `V̇₁V̇₂` or its cycle average
/// `⟨V̇₁V̇₂⟩`, depending on the caller. Returns 0 for non-positive
/// distance (no pair geometry).
fn crum_kernel(volume_rate_product: f64, distance: f64, density: f64) -> f64 {
    if distance <= 0.0 {
        return 0.0;
    }
    -density * volume_rate_product / (FOUR_PI * distance * distance)
}

/// Secondary Bjerknes force, instantaneous Crum (1975) form.
///
/// ```text
/// F_B(t) = −ρ₀ · V̇₁(t) · V̇₂(t) / (4πd²)
/// ```
///
/// where `V̇ᵢ = 4πRᵢ²Ṙᵢ` is the volumetric oscillation rate. The
/// returned scalar is the line-of-centres component of the force on
/// one bubble due to the other, positive away from the other bubble:
/// bubbles whose walls move in the same sense (both expanding or both
/// contracting, `V̇₁V̇₂ > 0`) attract (`F < 0`); opposite senses repel
/// (`F > 0`). Its cycle average over harmonic oscillation is
/// [`secondary_bjerknes_force_mean`] with
/// `⟨V̇₁V̇₂⟩ = ω²ΔV₁ΔV₂cos(φ)/2`.
///
/// # References
/// - Crum LA (1975). *J. Acoust. Soc. Am.* 57(2), 536–538.
/// - Leighton TG (1994). *The Acoustic Bubble*. Academic Press. §3.4.
/// - Brennen CE (1995). *Cavitation and Bubble Dynamics*. Eq. 4.49.
#[must_use]
pub fn secondary_bjerknes_force_instantaneous(
    volume_rate_1: f64,
    volume_rate_2: f64,
    distance: f64,
    density: f64,
) -> f64 {
    crum_kernel(volume_rate_1 * volume_rate_2, distance, density)
}

/// Secondary Bjerknes force from the cycle-averaged volume-rate
/// product: `F = −ρ₀ · ⟨V̇₁V̇₂⟩ / (4πd²)` (Crum 1975). Same sign
/// convention as [`secondary_bjerknes_force_instantaneous`]. For
/// harmonic oscillations with peak volume amplitudes `ΔVᵢ` and phase
/// difference `φ`, `⟨V̇₁V̇₂⟩ = ω²ΔV₁ΔV₂cos(φ)/2`, giving the
/// time-averaged form `F = −(ρ₀ω²/(8π)) · ΔV₁ΔV₂ · cos(φ) / d²` used
/// by [`BjerknesCalculator::secondary_bjerknes_force`].
#[must_use]
pub fn secondary_bjerknes_force_mean(
    mean_volume_rate_product: f64,
    distance: f64,
    density: f64,
) -> f64 {
    crum_kernel(mean_volume_rate_product, distance, density)
}

impl BjerknesCalculator {
    /// Calculate secondary Bjerknes force between two bubbles.
    ///
    /// ## Formula
    ///
    /// ```text
    /// F_B2 = −(ρ₀ω²/(8π)) · V₁ · V₂ · cos(φ) / d²
    /// ```
    ///
    /// The time-averaged acoustic radiation interaction
    /// (Bjerknes 1906; Crum 1975; Leighton 1994 §3.4):
    ///   F = −(ρ₀/(4πd²)) ⟨V̇₁V̇₂⟩
    /// For harmonic volume oscillations V_i(t) = V_i cos(ωt + φ_i):
    ///   ⟨V̇₁V̇₂⟩ = ω²V₁V₂cos(φ)/2  →  F = −(ρ₀ω²/(8π))·V₁V₂cos(φ)/d²,
    /// evaluated through the canonical kernel
    /// [`secondary_bjerknes_force_mean`].
    ///
    /// Sign convention (module docs): the scalar is the
    /// line-of-centres component of the force on one bubble due to
    /// the other, positive away from the other bubble:
    ///   cos φ > 0 (in phase)   → F < 0, attractive
    ///   cos φ < 0 (antiphase)  → F > 0, repulsive
    /// The `interaction_type` label follows the same physics
    /// (Attractive for in-phase pairs); it is not the sign of F.
    ///
    /// where:
    /// - V₁, V₂ are peak volume oscillation amplitudes (m³)
    /// - φ is phase difference between oscillations (rad)
    /// - d is centre-to-centre distance (m)
    ///
    /// ## References
    ///
    /// - Bjerknes V (1906). *Fields of Force*. Columbia UP.
    /// - Crum LA (1975). *J. Acoust. Soc. Am.* 57(2), 536–538.
    /// - Leighton TG (1994). *The Acoustic Bubble*. Academic Press. §3.4.
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn secondary_bjerknes_force(
        &self,
        bubble1_radius: f64,
        bubble2_radius: f64,
        volume_amplitude1: f64,
        volume_amplitude2: f64,
        phase_difference: f64,
        distance: f64,
    ) -> KwaversResult<BjerknesForceData> {
        if bubble1_radius <= 0.0 || bubble2_radius <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Bubble radii must be positive".to_owned(),
            ));
        }

        if distance <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Distance must be positive".to_owned(),
            ));
        }

        // Check if outside interaction range
        if distance > self.config.interaction_range {
            return Ok(BjerknesForceData {
                primary: 0.0,
                secondary: 0.0,
                total: 0.0,
                phase_difference,
                interaction_type: BjerknesInteractionType::Neutral,
                distance,
                coalescing: false,
            });
        }

        // Secondary Bjerknes force (Crum 1975; Leighton 1994, §3.4):
        //   F_B2 = −(ρ₀ω²/(8π)) · V₁ · V₂ · cos(φ) / d²
        // where ω = 2πf.  Derived from F = −(ρ₀/(4πd²))⟨V̇₁V̇₂⟩ with
        // V̇_i(t) = −V_i·ω·sin(ωt + φ_i), so ⟨V̇₁V̇₂⟩ = ω²V₁V₂cos(φ)/2.
        // Dimensional check: [kg/m³ · s⁻² · m³ · m³ / m²] = [kg·m/s²] = N ✓
        //
        // The signed cos(φ) carries the attractive (cos>0 → F<0) vs
        // repulsive (cos<0 → F>0) sense directly under the canonical
        // line-of-centres convention (module docs); the categorical
        // `interaction_type` below is a downstream label, not a gate
        // that zeroes the force.  Prior to 2026-05-21 this used
        // |cos φ| and zeroed the force whenever |cos φ| < 0.1,
        // producing a discontinuous step at φ = π/2 ± δ.  Prior to
        // the 2026-11-27 consolidation (KW-PHYS-009 item 1) the
        // returned scalar was the attraction-positive strength — the
        // negative of the Crum component computed here.
        let omega = TWO_PI * self.config.frequency;
        let cos_phase = phase_difference.cos();
        let mean_vdot_product =
            0.5 * omega * omega * volume_amplitude1 * volume_amplitude2 * cos_phase;

        let secondary_force =
            secondary_bjerknes_force_mean(mean_vdot_product, distance, self.config.rho);

        // Categorical label tracking the *sign* of the interaction.  The
        // ±0.1 dead-band is a labelling convenience; the force value above
        // is continuous through it.
        let interaction_type = if cos_phase > 0.1 {
            BjerknesInteractionType::Attractive
        } else if cos_phase < -0.1 {
            BjerknesInteractionType::Repulsive
        } else {
            BjerknesInteractionType::Neutral
        };

        // Check coalescence condition
        let coalescing = distance < self.config.coalescence_distance;

        let result = BjerknesForceData {
            primary: 0.0,
            secondary: secondary_force,
            total: secondary_force,
            phase_difference,
            interaction_type,
            distance,
            coalescing,
        };

        Ok(result)
    }

    /// Calculate combined primary and secondary Bjerknes forces
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    #[allow(clippy::too_many_arguments)]
    pub fn total_bjerknes_force(
        &self,
        bubble1_radius: f64,
        bubble2_radius: f64,
        acoustic_pressure: f64,
        pressure_gradient: f64,
        volume_amplitude1: f64,
        volume_amplitude2: f64,
        phase_difference: f64,
        distance: f64,
    ) -> KwaversResult<BjerknesForceData> {
        let mut result = self.secondary_bjerknes_force(
            bubble1_radius,
            bubble2_radius,
            volume_amplitude1,
            volume_amplitude2,
            phase_difference,
            distance,
        )?;

        if self.config.include_primary {
            let primary =
                self.primary_bjerknes_force(bubble1_radius, acoustic_pressure, pressure_gradient)?;
            result.primary = primary;
            result.total = result.primary + result.secondary;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        calculator::BjerknesCalculator,
        types::{BjerknesConfig, BjerknesInteractionType},
    };
    use std::f64::consts::PI;

    fn calc() -> BjerknesCalculator {
        BjerknesCalculator::new(BjerknesConfig::default())
    }

    /// Non-positive radii or distance → Err.
    #[test]
    fn secondary_bjerknes_force_errors_for_invalid_geometry() {
        let c = calc();
        assert!(c
            .secondary_bjerknes_force(0.0, 5e-6, 1e-15, 1e-15, 0.0, 10e-6)
            .is_err());
        assert!(c
            .secondary_bjerknes_force(5e-6, 0.0, 1e-15, 1e-15, 0.0, 10e-6)
            .is_err());
        assert!(c
            .secondary_bjerknes_force(5e-6, 5e-6, 1e-15, 1e-15, 0.0, 0.0)
            .is_err());
        assert!(c
            .secondary_bjerknes_force(5e-6, 5e-6, 1e-15, 1e-15, 0.0, -1e-6)
            .is_err());
    }

    /// Phase=0 (in-phase) → attractive interaction.
    ///
    /// cos(0) = 1 > 0.1 → BjerknesInteractionType::Attractive. Under
    /// the canonical line-of-centres convention (KW-PHYS-009 item 1
    /// consolidation, 2026-11-27) attraction is a NEGATIVE scalar:
    /// the force on each bubble points toward the other. (Pre-
    /// consolidation this asserted the old attraction-positive sign.)
    #[test]
    fn secondary_bjerknes_force_attractive_for_zero_phase_difference() {
        let result = calc()
            .secondary_bjerknes_force(5e-6, 5e-6, 1e-15, 1e-15, 0.0, 10e-6)
            .unwrap();
        assert_eq!(result.interaction_type, BjerknesInteractionType::Attractive);
        assert!(
            result.secondary < 0.0,
            "in-phase (attractive) force must be negative under the \
             line-of-centres convention, got {}",
            result.secondary
        );
    }

    /// Phase=π (anti-phase) → repulsive interaction.
    ///
    /// cos(π) = −1 < −0.1 → BjerknesInteractionType::Repulsive, and
    /// the scalar is positive (directed away from the other bubble).
    #[test]
    fn secondary_bjerknes_force_repulsive_for_pi_phase_difference() {
        let result = calc()
            .secondary_bjerknes_force(5e-6, 5e-6, 1e-15, 1e-15, PI, 10e-6)
            .unwrap();
        assert_eq!(result.interaction_type, BjerknesInteractionType::Repulsive);
        assert!(
            result.secondary > 0.0,
            "antiphase (repulsive) force must be positive under the \
             line-of-centres convention, got {}",
            result.secondary
        );
    }

    /// Canonical kernel: same-sense volume rates attract (F < 0),
    /// opposite senses repel (F > 0), non-positive distance → 0.
    #[test]
    fn canonical_kernel_signs_follow_crum_convention() {
        let rho = 1000.0;
        let d = 50e-6;
        // Both expanding (V̇₁V̇₂ > 0): attraction.
        assert!(
            super::secondary_bjerknes_force_instantaneous(1e-9, 2e-9, d, rho) < 0.0,
            "in-phase volume rates must attract"
        );
        // Both contracting (product still > 0): attraction.
        assert!(
            super::secondary_bjerknes_force_instantaneous(-1e-9, -2e-9, d, rho) < 0.0,
            "in-phase (contracting) volume rates must attract"
        );
        // Opposite senses: repulsion.
        assert!(
            super::secondary_bjerknes_force_instantaneous(1e-9, -2e-9, d, rho) > 0.0,
            "antiphase volume rates must repel"
        );
        assert_eq!(
            super::secondary_bjerknes_force_instantaneous(1e-9, 2e-9, 0.0, rho),
            0.0
        );
        assert_eq!(
            super::secondary_bjerknes_force_mean(1e-18, -1.0, rho),
            0.0,
            "mean form shares the kernel's distance guard"
        );
    }

    /// The calculator's harmonic form is the canonical kernel applied
    /// to ⟨V̇₁V̇₂⟩ = ω²ΔV₁ΔV₂cos(φ)/2 — the two public forms of the
    /// law agree exactly (KW-PHYS-009 item 1 consolidation).
    #[test]
    fn calculator_matches_canonical_mean_kernel() {
        let c = calc();
        let cfg = c.config();
        let omega = std::f64::consts::TAU * cfg.frequency;
        let (v1, v2, d) = (1e-15, 3e-15, 40e-6);
        for phase in [0.0, 0.7, PI, 4.0] {
            let via_calculator = c
                .secondary_bjerknes_force(5e-6, 5e-6, v1, v2, phase, d)
                .unwrap()
                .secondary;
            let mean_product = 0.5 * omega * omega * v1 * v2 * phase.cos();
            let via_kernel = super::secondary_bjerknes_force_mean(mean_product, d, cfg.rho);
            assert!(
                (via_calculator - via_kernel).abs() <= 1e-12 * via_kernel.abs().max(1e-300),
                "calculator ({via_calculator:e}) must equal canonical kernel ({via_kernel:e}) at phase {phase}"
            );
        }
    }

    /// Distance beyond interaction_range → all forces zero, Neutral.
    #[test]
    fn secondary_bjerknes_force_zero_beyond_interaction_range() {
        let cfg = BjerknesConfig {
            interaction_range: 50e-6,
            ..BjerknesConfig::default()
        };
        let c = BjerknesCalculator::new(cfg);
        let result = c
            .secondary_bjerknes_force(5e-6, 5e-6, 1e-15, 1e-15, 0.0, 100e-6)
            .unwrap();
        assert_eq!(
            result.secondary, 0.0,
            "force must be 0 beyond interaction range"
        );
        assert_eq!(result.interaction_type, BjerknesInteractionType::Neutral);
    }

    /// Force must be continuous through cos(φ) = ±0.1 (the categorical
    /// "Neutral" boundary).  Regression test for the pre-2026-05-21 bug where
    /// the implementation zeroed the force whenever |cos φ| < 0.1, producing
    /// a discontinuous step.
    #[test]
    fn secondary_bjerknes_force_continuous_through_neutral_threshold() {
        let c = calc();
        // Sample cos(φ) just above and just below the 0.1 threshold.
        let phi_above = (0.101_f64).acos();
        let phi_below = (0.099_f64).acos();
        let r1 = 5e-6;
        let r2 = 5e-6;
        let v = 1e-15;
        let d = 10e-6;
        let f_above = c
            .secondary_bjerknes_force(r1, r2, v, v, phi_above, d)
            .unwrap();
        let f_below = c
            .secondary_bjerknes_force(r1, r2, v, v, phi_below, d)
            .unwrap();
        // Forces must differ by ≲ 2 % across the 0.1 / 0.099 step (linear in cos).
        let rel_diff =
            (f_above.secondary - f_below.secondary).abs() / f_above.secondary.abs().max(1e-300);
        assert!(
            rel_diff < 0.05,
            "secondary Bjerknes force must be continuous through cos(φ)=0.1 \
             (got f_above={fa:.3e}, f_below={fb:.3e}, rel_diff={rel_diff:.3})",
            fa = f_above.secondary,
            fb = f_below.secondary,
        );
        assert!(
            f_below.secondary != 0.0,
            "f_below must be nonzero — was {fb}",
            fb = f_below.secondary
        );
    }

    /// Coalescence flag set when distance < coalescence_distance.
    #[test]
    fn secondary_bjerknes_force_sets_coalescence_flag_below_threshold() {
        let cfg = BjerknesConfig {
            coalescence_distance: 5e-6,
            interaction_range: 100e-6,
            ..BjerknesConfig::default()
        };
        let c = BjerknesCalculator::new(cfg);
        let near = c
            .secondary_bjerknes_force(5e-6, 5e-6, 1e-15, 1e-15, 0.0, 2e-6)
            .unwrap();
        let far = c
            .secondary_bjerknes_force(5e-6, 5e-6, 1e-15, 1e-15, 0.0, 20e-6)
            .unwrap();
        assert!(
            near.coalescing,
            "must coalesce when d < coalescence_distance"
        );
        assert!(
            !far.coalescing,
            "must not coalesce when d > coalescence_distance"
        );
    }
}
