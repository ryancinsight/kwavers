//! Secondary Bjerknes force calculations

use super::calculator::BjerknesCalculator;
use super::types::{BjerknesForceData, BjerknesInteractionType};
use crate::core::error::{KwaversError, KwaversResult};
use crate::core::constants::numerical::{TWO_PI};

impl BjerknesCalculator {
    /// Calculate secondary Bjerknes force between two bubbles.
    ///
    /// ## Formula
    ///
    /// ```text
    /// F_B2 = (ρ₀ω²/(8π)) · V₁ · V₂ · cos(φ) / d²
    /// ```
    ///
    /// Derived from the time-averaged acoustic radiation interaction
    /// (Bjerknes 1906; Leighton 1994 §3.4):
    ///   F = −(ρ₀/(4πd²)) ⟨V̇₁V̇₂⟩
    /// For harmonic volume oscillations V_i(t) = V_i cos(ωt + φ_i):
    ///   ⟨V̇₁V̇₂⟩ = ω²V₁V₂cos(φ)/2  → coefficient = ρ₀ω²/(8π)
    ///
    /// Sign convention (scalar interaction strength):
    ///   cos φ > 0  → attractive (positive)
    ///   cos φ < 0  → repulsive  (negative)
    ///
    /// where:
    /// - V₁, V₂ are peak volume oscillation amplitudes (m³)
    /// - φ is phase difference between oscillations (rad)
    /// - d is centre-to-centre distance (m)
    ///
    /// ## References
    ///
    /// - Bjerknes V (1906). *Fields of Force*. Columbia UP.
    /// - Leighton TG (1994). *The Acoustic Bubble*. Academic Press. §3.4.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
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

        // Secondary Bjerknes force (Leighton 1994, §3.4; Bjerknes 1906):
        //   F_B2 = (ρ₀ω²/(8π)) · V₁ · V₂ · cos(φ) / d²
        // where ω = 2πf.  Derived from F = −(ρ₀/(4πd²))⟨V̇₁V̇₂⟩ with
        // V̇_i(t) = V_i·ω·sin(ωt + φ_i) gives ⟨V̇₁V̇₂⟩ = ω²V₁V₂cos(φ)/2.
        // Dimensional check: [kg/m³ · s⁻² · m³ · m³ / m²] = [kg·m/s²] = N ✓
        //
        // The signed cos(φ) carries the attractive (cos>0) vs repulsive
        // (cos<0) sense directly; the categorical `interaction_type` below
        // is a downstream label, not a gate that zeroes the force.  Prior
        // to 2026-05-21 this used |cos φ| and zeroed the force whenever
        // |cos φ| < 0.1, producing a discontinuous step at φ = π/2 ± δ.
        let omega = TWO_PI * self.config.frequency;
        let coefficient = (self.config.rho * omega * omega) / (8.0 * std::f64::consts::PI);
        let cos_phase = phase_difference.cos();

        let secondary_force =
            coefficient * volume_amplitude1 * volume_amplitude2 * cos_phase / distance.powi(2);

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
    /// - Propagates any [`KwaversError`] returned by called functions.
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
    /// cos(0) = 1 > 0.1 → BjerknesInteractionType::Attractive.
    #[test]
    fn secondary_bjerknes_force_attractive_for_zero_phase_difference() {
        let result = calc()
            .secondary_bjerknes_force(5e-6, 5e-6, 1e-15, 1e-15, 0.0, 10e-6)
            .unwrap();
        assert_eq!(result.interaction_type, BjerknesInteractionType::Attractive);
        assert!(result.secondary > 0.0, "attractive force must be positive");
    }

    /// Phase=π (anti-phase) → repulsive interaction.
    ///
    /// cos(π) = −1 < −0.1 → BjerknesInteractionType::Repulsive.
    #[test]
    fn secondary_bjerknes_force_repulsive_for_pi_phase_difference() {
        let result = calc()
            .secondary_bjerknes_force(5e-6, 5e-6, 1e-15, 1e-15, PI, 10e-6)
            .unwrap();
        assert_eq!(result.interaction_type, BjerknesInteractionType::Repulsive);
        assert!(result.secondary < 0.0, "repulsive force must be negative");
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
