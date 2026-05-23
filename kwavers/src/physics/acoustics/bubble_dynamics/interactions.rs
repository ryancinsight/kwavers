//! Bubble-bubble interactions
//!
//! Calculates forces between bubbles (Bjerknes forces, etc.)

use super::bubble_state::BubbleState;
use crate::core::constants::fundamental::{ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL};
use ndarray::Array3;
use std::collections::HashMap;

/// Bubble interaction calculator
#[derive(Debug)]
pub struct BubbleInteractions {
    /// Interaction cutoff distance \[m\]
    pub cutoff_distance: f64,
    /// Interaction strength factor
    pub interaction_strength: f64,
    /// Liquid density \[kg/m³\] — used in the monopole pressure formula
    pub liquid_density: f64,
}

impl Default for BubbleInteractions {
    fn default() -> Self {
        Self {
            cutoff_distance: 1e-3, // 1 mm
            interaction_strength: 1.0,
            liquid_density: DENSITY_WATER_NOMINAL,
        }
    }
}

impl BubbleInteractions {
    /// Calculate interaction pressure field from all bubbles
    #[must_use]
    pub fn calculate_interaction_field(
        &self,
        bubbles: &HashMap<(usize, usize, usize), BubbleState>,
        grid_shape: (usize, usize, usize),
        grid_spacing: (f64, f64, f64),
    ) -> Array3<f64> {
        let mut interaction_field = Array3::zeros(grid_shape);

        // For each grid point
        ndarray::Zip::indexed(&mut interaction_field).par_for_each(|(i, j, k), field_val| {
            let total_interaction = bubbles
                .iter()
                .filter(|((bi, bj, bk), _)| !(i == *bi && j == *bj && k == *bk)) // Skip self-interaction
                .filter_map(|((bi, bj, bk), state)| {
                    // Calculate distance
                    let dx = (i as f64 - *bi as f64) * grid_spacing.0;
                    let dy = (j as f64 - *bj as f64) * grid_spacing.1;
                    let dz = (k as f64 - *bk as f64) * grid_spacing.2;
                    let distance = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();

                    if distance < self.cutoff_distance && distance > 0.0 {
                        // Bjerknes force contribution
                        Some(self.calculate_bjerknes_contribution(state, distance))
                    } else {
                        None
                    }
                })
                .sum::<f64>();

            *field_val = total_interaction * self.interaction_strength;
        });

        interaction_field
    }

    /// Acoustic pressure contribution at `distance` from a pulsating bubble.
    ///
    /// For a spherical bubble treated as an acoustic monopole the far-field
    /// radiated pressure is (Leighton 1994, §3.3; Brennen 1995 Eq. 2.57):
    ///
    /// ```text
    /// p(r, t) = ρ · V̈(t) / (4π · r)
    /// ```
    ///
    /// where `V̈ = dV̇/dt = 4π R² R̈ + 8π R Ṙ²` is the volumetric
    /// acceleration, `r` is the observation distance, and `ρ` is the liquid
    /// density.
    ///
    /// Dimensional analysis: \[kg/m³ · m³/s² / m\] = \[Pa\] ✓.
    ///
    /// The previous implementation returned `V̇/(4π·r²) + V·R̈/r²` which has
    /// units of m/s — a factor of ρ·r short of pressure — and used 1/r²
    /// instead of the physically correct 1/r near-field approximation.
    fn calculate_bjerknes_contribution(&self, bubble: &BubbleState, distance: f64) -> f64 {
        // V̈ = d(V̇)/dt = d(4π R² Ṙ)/dt = 4π R² R̈ + 8π R Ṙ²
        let r = bubble.radius;
        let r_dot = bubble.wall_velocity;
        let r_ddot = bubble.wall_acceleration;
        let v_ddot = 4.0 * std::f64::consts::PI * r * r * r_ddot
            + 8.0 * std::f64::consts::PI * r * r_dot * r_dot;

        // Monopole acoustic pressure: p = ρ · V̈ / (4π · r)  [Pa]
        self.liquid_density * v_ddot / (4.0 * std::f64::consts::PI * distance)
    }
}

/// Calculate Bjerknes force between two bubbles
#[derive(Debug)]
pub struct BjerknesForceComputer;

impl BjerknesForceComputer {
    /// Primary Bjerknes force (bubble in pressure gradient)
    #[must_use]
    pub fn primary(bubble_volume: f64, pressure_gradient: f64) -> f64 {
        -bubble_volume * pressure_gradient
    }

    /// Secondary Bjerknes force (bubble-bubble interaction).
    ///
    /// Canonical instantaneous form (Crum 1975; Leighton 1994 §4.5.2;
    /// Brennen 1995 Eq. 4.49):
    ///
    /// ```text
    /// F_B(t) = - ρ₀ V̇₁(t) V̇₂(t) / (4π d²)
    /// ```
    ///
    /// where V̇_i = 4π R_i² Ṙ_i is the volumetric oscillation rate. The
    /// inverse-square distance dependence is required by the integrability
    /// of the radial Bernoulli pressure field around a pulsating monopole;
    /// any other power produces an incorrect far-field decay.
    ///
    /// Dimensional check: \[kg/m³ · (m³/s)² / m²\] = \[kg·m/s²\] = N ✓.
    #[must_use]
    pub fn secondary(
        bubble1: &BubbleState,
        bubble2: &BubbleState,
        distance: f64,
        liquid_density: f64,
    ) -> f64 {
        if distance <= 0.0 {
            return 0.0;
        }

        let v1_dot = 4.0 * std::f64::consts::PI * bubble1.radius.powi(2) * bubble1.wall_velocity;
        let v2_dot = 4.0 * std::f64::consts::PI * bubble2.radius.powi(2) * bubble2.wall_velocity;

        -liquid_density * v1_dot * v2_dot / (4.0 * std::f64::consts::PI * distance.powi(2))
    }

    /// Check if bubbles attract or repel
    #[must_use]
    pub fn interaction_type(bubble1: &BubbleState, bubble2: &BubbleState) -> BubbleInteractionType {
        // In phase: both expanding or both contracting -> attraction
        // Out of phase: one expanding, one contracting -> repulsion

        if bubble1.wall_velocity * bubble2.wall_velocity > 0.0 {
            BubbleInteractionType::Attraction
        } else {
            BubbleInteractionType::Repulsion
        }
    }
}

/// Type of bubble-bubble interaction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BubbleInteractionType {
    Attraction,
    Repulsion,
    Neutral,
}

/// Collective effects in bubble clouds
#[derive(Debug)]
pub struct CollectiveEffects;

impl CollectiveEffects {
    /// Calculate effective sound speed in bubbly liquid
    #[must_use]
    pub fn wood_sound_speed(
        void_fraction: f64,
        liquid_density: f64,
        liquid_sound_speed: f64,
        gas_density: f64,
        gas_sound_speed: f64,
    ) -> f64 {
        // Wood's equation for sound speed in bubbly liquid
        let mixture_density =
            (1.0 - void_fraction).mul_add(liquid_density, void_fraction * gas_density);
        let compressibility = (1.0 - void_fraction) / (liquid_density * liquid_sound_speed.powi(2))
            + void_fraction / (gas_density * gas_sound_speed.powi(2));

        (mixture_density * compressibility).powf(-0.5)
    }

    /// Calculate void fraction from bubble field
    #[must_use]
    pub fn void_fraction(
        bubbles: &HashMap<(usize, usize, usize), BubbleState>,
        grid_volume: f64,
    ) -> f64 {
        let total_bubble_volume: f64 = bubbles
            .values()
            .map(super::bubble_state::BubbleState::volume)
            .sum();

        total_bubble_volume / grid_volume
    }

    /// Estimate collective oscillation frequency
    #[must_use]
    pub fn collective_frequency(
        mean_radius: f64,
        void_fraction: f64,
        liquid_properties: (f64, f64), // (density, sound_speed)
    ) -> f64 {
        let (rho, _c) = liquid_properties;

        // Minnaert frequency modified for void fraction
        let f0 = 1.0 / (2.0 * std::f64::consts::PI * mean_radius)
            * (3.0 * 1.4 * ATMOSPHERIC_PRESSURE / rho).sqrt();

        // Correction for bubble-bubble interactions
        f0 * (1.0 - void_fraction).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
    use crate::physics::bubble_dynamics::bubble_state::BubbleParameters;

    #[test]
    fn test_bjerknes_interaction_type() {
        let params = BubbleParameters::default();
        let mut bubble1 = BubbleState::new(&params);
        let mut bubble2 = BubbleState::new(&params);

        // Both expanding
        bubble1.wall_velocity = 10.0;
        bubble2.wall_velocity = 5.0;
        assert_eq!(
            BjerknesForceComputer::interaction_type(&bubble1, &bubble2),
            BubbleInteractionType::Attraction
        );

        // Opposite phase
        bubble1.wall_velocity = 10.0;
        bubble2.wall_velocity = -5.0;
        assert_eq!(
            BjerknesForceComputer::interaction_type(&bubble1, &bubble2),
            BubbleInteractionType::Repulsion
        );
    }

    #[test]
    fn secondary_bjerknes_scales_as_inverse_square_distance() {
        // F_B ∝ 1/d² ⇒ F(d)·d² is independent of d when V̇₁,V̇₂,ρ are fixed.
        let params = BubbleParameters::default();
        let mut bubble1 = BubbleState::new(&params);
        let mut bubble2 = BubbleState::new(&params);
        bubble1.radius = 2.0e-6;
        bubble2.radius = 3.0e-6;
        bubble1.wall_velocity = 0.5;
        bubble2.wall_velocity = -0.4;
        let rho = DENSITY_WATER_NOMINAL;

        let d_a = 1.0e-4_f64;
        let d_b = 4.0e-4_f64;
        let f_a = BjerknesForceComputer::secondary(&bubble1, &bubble2, d_a, rho);
        let f_b = BjerknesForceComputer::secondary(&bubble1, &bubble2, d_b, rho);

        // Both should be positive (out-of-phase, so -ρ·V̇₁·V̇₂·(...) flips sign positive).
        assert!(f_a.abs() > 0.0);
        // Ratio must equal (d_b/d_a)² = 16 — failing here detected the previous
        // /distance (1/d) bug, which would have produced ratio 4.
        let ratio = f_a / f_b;
        let expected = (d_b / d_a).powi(2);
        assert!(
            (ratio - expected).abs() / expected < 1.0e-12,
            "secondary Bjerknes ratio {ratio} should equal {expected} = (d_b/d_a)²"
        );
    }

    #[test]
    fn test_wood_sound_speed() {
        let c_mixture = CollectiveEffects::wood_sound_speed(
            0.01,                  // 1% void fraction
            1000.0,                // water density
            SOUND_SPEED_WATER_SIM, // water sound speed
            1.2,                   // air density
            340.0,                 // air sound speed
        );

        // Sound speed should be significantly reduced
        assert!(c_mixture < SOUND_SPEED_WATER_SIM);
        assert!(c_mixture > 100.0); // But still reasonable
    }
}
