//! Secondary Bjerknes force calculations

use super::calculator::BjerknesCalculator;
use super::types::{BjerknesForce, InteractionType};
use crate::core::error::{KwaversError, KwaversResult};

impl BjerknesCalculator {
    /// Calculate secondary Bjerknes force between two bubbles
    ///
    /// Interaction force between bubble pair due to mutual acoustic scattering:
    /// F_s ≈ (6πρc₀/f) * V₁ * V₂ * cos(φ) / d⁴
    ///
    /// where:
    /// - V₁, V₂ are volume oscillation amplitudes
    /// - φ is phase difference between oscillations
    /// - d is center-to-center distance
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
    ) -> KwaversResult<BjerknesForce> {
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
            return Ok(BjerknesForce {
                primary: 0.0,
                secondary: 0.0,
                total: 0.0,
                phase_difference,
                interaction_type: InteractionType::Neutral,
                distance,
                coalescing: false,
            });
        }

        // Secondary Bjerknes force magnitude
        // F_s = (6πρc₀/f) * V₁ * V₂ * |cos(φ)| / d⁴
        let coefficient =
            (6.0 * std::f64::consts::PI * self.config.rho * self.config.c0) / self.config.frequency;

        let force_magnitude =
            coefficient * volume_amplitude1 * volume_amplitude2 * phase_difference.cos().abs()
                / distance.powi(4);

        // Determine interaction type based on phase difference
        let cos_phase = phase_difference.cos();
        let interaction_type = if cos_phase > 0.1 {
            InteractionType::Attractive
        } else if cos_phase < -0.1 {
            InteractionType::Repulsive
        } else {
            InteractionType::Neutral
        };

        // Apply sign based on interaction type
        let secondary_force = match interaction_type {
            InteractionType::Attractive => force_magnitude, // Positive (toward each other)
            InteractionType::Repulsive => -force_magnitude, // Negative (away)
            InteractionType::Neutral => 0.0,
        };

        // Check coalescence condition
        let coalescing = distance < self.config.coalescence_distance;

        let result = BjerknesForce {
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
    ) -> KwaversResult<BjerknesForce> {
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
        types::{BjerknesConfig, InteractionType},
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
    /// cos(0) = 1 > 0.1 → InteractionType::Attractive.
    #[test]
    fn secondary_bjerknes_force_attractive_for_zero_phase_difference() {
        let result = calc()
            .secondary_bjerknes_force(5e-6, 5e-6, 1e-15, 1e-15, 0.0, 10e-6)
            .unwrap();
        assert_eq!(result.interaction_type, InteractionType::Attractive);
        assert!(result.secondary > 0.0, "attractive force must be positive");
    }

    /// Phase=π (anti-phase) → repulsive interaction.
    ///
    /// cos(π) = −1 < −0.1 → InteractionType::Repulsive.
    #[test]
    fn secondary_bjerknes_force_repulsive_for_pi_phase_difference() {
        let result = calc()
            .secondary_bjerknes_force(5e-6, 5e-6, 1e-15, 1e-15, PI, 10e-6)
            .unwrap();
        assert_eq!(result.interaction_type, InteractionType::Repulsive);
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
        assert_eq!(result.interaction_type, InteractionType::Neutral);
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
