//! Bjerknes Forces: Bubble-Bubble Interactions in Acoustic Fields
//!
//! This module implements primary and secondary Bjerknes forces that govern
//! the interactions between microbubbles in acoustic fields.
//!
//! ## Physical Background
//!
//! **Primary Bjerknes Force**:
//! Arises from the time-averaged pressure gradient in the acoustic field acting on
//! the oscillating bubble surface. Each bubble experiences a radiation force.
//!
//! ```
//! F_primary = -4πR² ⟨p'(∂u'/∂z)⟩
//! ```
//! where ⟨⟩ denotes time averaging and R is bubble radius
//!
//! **Secondary Bjerknes Force**:
//! Results from the acoustic field scattered by one bubble affecting another.
//! This force can be attractive (bubbles oscillate in phase) or repulsive
//! (out of phase oscillations).
//!
//! ```
//! F_secondary ≈ (6πρc₀/f) * V₁ * V₂ * cos(φ) / d²
//! ```
//! where V₁, V₂ are bubble volume amplitudes, φ is phase difference, d is separation
//!
//! ## Physical Effects
//!
//! - **Attraction**: In-phase oscillations → secondary force attractive
//! - **Repulsion**: Out-of-phase oscillations → secondary force repulsive
//! - **Coalescence**: When bubbles approach within coalescence distance
//! - **Fragmentation**: Large bubbles break due to acoustic stress
//!
//! ## References
//!
//! - Björknes, V. (1906). "Fields of force" Columbia University Press
//! - Crum, L. A. (1975). "Bjerknes forces on bubbles in a stationary sound field"
//! - Pozarowski, P. & Holyst, R. (2002) "Two and three body interactions"
//! - Garbin, V. et al. (2007) "Dynamics of coated bubbles monitored by individual pulse acoustography"

use crate::core::error::{KwaversError, KwaversResult};
use std::f64;

/// Configuration for Bjerknes force calculations
#[derive(Debug, Clone, Copy)]
pub struct BjerknesConfig {
    /// Sound speed in medium (m/s)
    pub c0: f64,

    /// Medium density (kg/m³)
    pub rho: f64,

    /// Operating frequency (Hz)
    pub frequency: f64,

    /// Enable primary Bjerknes force
    pub include_primary: bool,

    /// Enable secondary Bjerknes force
    pub include_secondary: bool,

    /// Coalescence threshold distance (m)
    pub coalescence_distance: f64,

    /// Maximum interaction distance (m)
    pub interaction_range: f64,
}

impl Default for BjerknesConfig {
    fn default() -> Self {
        Self {
            c0: 1540.0,
            rho: 1000.0,
            frequency: 1e6,
            include_primary: true,
            include_secondary: true,
            coalescence_distance: 1e-6, // 1 μm
            interaction_range: 100e-6,  // 100 μm
        }
    }
}

/// Results from Bjerknes force calculation
#[derive(Debug, Clone, Copy)]
pub struct BjerknesForce {
    /// Primary Bjerknes force (N) - radiation pressure force
    pub primary: f64,

    /// Secondary Bjerknes force (N) - bubble-bubble interaction
    pub secondary: f64,

    /// Total force (N)
    pub total: f64,

    /// Phase difference between bubbles (radians)
    pub phase_difference: f64,

    /// Interaction type (attractive/repulsive)
    pub interaction_type: InteractionType,

    /// Distance between bubbles (m)
    pub distance: f64,

    /// Whether bubbles will coalesce
    pub coalescing: bool,
}

/// Type of interaction between bubbles
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionType {
    /// Attractive interaction (bubbles approach)
    Attractive,

    /// Repulsive interaction (bubbles separate)
    Repulsive,

    /// No significant interaction
    Neutral,
}

/// Bjerknes force calculator for bubble-bubble interactions
#[derive(Debug)]
pub struct BjerknesCalculator {
    config: BjerknesConfig,
}

impl BjerknesCalculator {
    /// Create new Bjerknes force calculator
    pub fn new(config: BjerknesConfig) -> Self {
        Self { config }
    }

    /// Calculate primary Bjerknes force on a bubble in acoustic field
    ///
    /// The primary Bjerknes force arises from the time-averaged radiation pressure:
    /// F_p = -4πR² ⟨p'(∂u'/∂z)⟩
    ///
    /// For a plane wave: F_p ≈ πR² I / c₀
    /// where I is acoustic intensity
    pub fn primary_bjerknes_force(
        &self,
        bubble_radius: f64,
        acoustic_pressure_amplitude: f64,
        pressure_gradient: f64,
    ) -> KwaversResult<f64> {
        if bubble_radius <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Bubble radius must be positive".to_string(),
            ));
        }

        // Primary Bjerknes force for acoustic wave
        // F_p = -4πR² ⟨p'(∂u'/∂z)⟩
        // Approximation: F_p ≈ -4πR² (p₀/(2ρc₀)) (∂p/∂z)
        // where the time-averaged acoustic intensity contribution is proportional to pressure amplitude and gradient

        let surface_area = 4.0 * f64::consts::PI * bubble_radius.powi(2);

        // Acoustic radiation pressure contribution
        let radiation_pressure =
            acoustic_pressure_amplitude.powi(2) / (2.0 * self.config.rho * self.config.c0);

        // Force from pressure gradient
        let force = -surface_area * radiation_pressure * (pressure_gradient / self.config.c0);

        Ok(force)
    }

    /// Calculate secondary Bjerknes force between two bubbles
    ///
    /// Interaction force between bubble pair due to mutual acoustic scattering:
    /// F_s ≈ (6πρc₀/f) * V₁ * V₂ * cos(φ) / d⁴
    ///
    /// where:
    /// - V₁, V₂ are volume oscillation amplitudes
    /// - φ is phase difference between oscillations
    /// - d is center-to-center distance
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
                "Bubble radii must be positive".to_string(),
            ));
        }

        if distance <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Distance must be positive".to_string(),
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
            (6.0 * f64::consts::PI * self.config.rho * self.config.c0) / self.config.frequency;

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

    /// Predict bubble trajectory under Bjerknes forces
    ///
    /// Simple model: acceleration = Force / mass
    /// where mass ≈ 3 * ρ * (4/3)πR³ for bubble in fluid
    pub fn predict_bubble_motion(
        &self,
        bubble_radius: f64,
        bjerknes_force: f64,
        initial_velocity: f64,
        time_step: f64,
    ) -> KwaversResult<(f64, f64)> {
        if bubble_radius <= 0.0 || time_step <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Bubble radius and time step must be positive".to_string(),
            ));
        }

        // Effective mass (including added mass effect)
        let bubble_volume = (4.0 / 3.0) * f64::consts::PI * bubble_radius.powi(3);
        let effective_mass = 3.0 * self.config.rho * bubble_volume; // With added mass factor

        // Calculate acceleration
        let acceleration = bjerknes_force / effective_mass;

        // Update velocity and position (simple Euler integration)
        let new_velocity = initial_velocity + acceleration * time_step;
        let displacement = initial_velocity * time_step + 0.5 * acceleration * time_step.powi(2);

        Ok((displacement, new_velocity))
    }

    /// Calculate coalescence probability based on approach distance
    pub fn coalescence_probability(&self, distance: f64, approach_velocity: f64) -> f64 {
        if distance > self.config.coalescence_distance {
            return 0.0; // Too far apart
        }

        if approach_velocity <= 0.0 {
            return 0.0; // Moving apart
        }

        // Simple model: probability increases as bubbles get closer
        // and approach velocity increases
        let proximity_factor = 1.0 - (distance / self.config.coalescence_distance);
        let velocity_factor = (approach_velocity / 1.0).min(1.0); // Normalize to 1.0 for 1 m/s

        (proximity_factor * velocity_factor).min(1.0)
    }

    /// Get configuration
    pub fn config(&self) -> BjerknesConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bjerknes_calculator_creation() {
        let config = BjerknesConfig::default();
        let _calc = BjerknesCalculator::new(config);
    }

    #[test]
    fn test_primary_bjerknes_force_calculation() {
        let config = BjerknesConfig::default();
        let calc = BjerknesCalculator::new(config);

        let radius = 5e-6; // 5 μm
        let pressure = 100e3; // 100 kPa
        let gradient = 1e6; // 1 MPa/m

        let force = calc
            .primary_bjerknes_force(radius, pressure, gradient)
            .unwrap();

        assert!(force.is_finite());
    }

    #[test]
    fn test_secondary_bjerknes_attractive() {
        let config = BjerknesConfig::default();
        let calc = BjerknesCalculator::new(config);

        let r1 = 5e-6;
        let r2 = 5e-6;
        let v1 = 1e-15;
        let v2 = 1e-15;
        let phase = 0.0; // In-phase oscillations
        let distance = 50e-6;

        let force = calc
            .secondary_bjerknes_force(r1, r2, v1, v2, phase, distance)
            .unwrap();

        assert_eq!(force.interaction_type, InteractionType::Attractive);
        assert!(force.coalescing == false); // Distance > coalescence distance
    }

    #[test]
    fn test_secondary_bjerknes_repulsive() {
        let config = BjerknesConfig::default();
        let calc = BjerknesCalculator::new(config);

        let r1 = 5e-6;
        let r2 = 5e-6;
        let v1 = 1e-15;
        let v2 = 1e-15;
        let phase = f64::consts::PI; // Out-of-phase oscillations
        let distance = 50e-6;

        let force = calc
            .secondary_bjerknes_force(r1, r2, v1, v2, phase, distance)
            .unwrap();

        assert_eq!(force.interaction_type, InteractionType::Repulsive);
    }

    #[test]
    fn test_coalescence_detection() {
        let config = BjerknesConfig::default();
        let calc = BjerknesCalculator::new(config);

        let r1 = 5e-6;
        let r2 = 5e-6;
        let v1 = 1e-15;
        let v2 = 1e-15;
        let phase = 0.0;

        // Very close distance (within coalescence range)
        let distance = 0.5e-6; // 0.5 μm
        let force = calc
            .secondary_bjerknes_force(r1, r2, v1, v2, phase, distance)
            .unwrap();

        assert!(force.coalescing);
    }

    #[test]
    fn test_bubble_motion_prediction() {
        let config = BjerknesConfig::default();
        let calc = BjerknesCalculator::new(config);

        let radius = 5e-6;
        let force = 1e-12; // 1 pN
        let velocity = 0.1; // 0.1 m/s
        let dt = 1e-6; // 1 μs

        let result = calc
            .predict_bubble_motion(radius, force, velocity, dt)
            .unwrap();

        assert!(result.0.is_finite()); // Displacement
        assert!(result.1.is_finite()); // Velocity
    }

    #[test]
    fn test_coalescence_probability() {
        let config = BjerknesConfig::default();
        let calc = BjerknesCalculator::new(config);

        // Close distance, approaching
        let prob_approach = calc.coalescence_probability(0.5e-6, 0.1);
        assert!(prob_approach > 0.0);

        // Far distance
        let prob_far = calc.coalescence_probability(100e-6, 0.1);
        assert_eq!(prob_far, 0.0);

        // Moving apart
        let prob_separate = calc.coalescence_probability(0.5e-6, -0.1);
        assert_eq!(prob_separate, 0.0);
    }

    #[test]
    fn test_interaction_range() {
        let config = BjerknesConfig::default();
        let calc = BjerknesCalculator::new(config);

        let r1 = 5e-6;
        let r2 = 5e-6;
        let v1 = 1e-15;
        let v2 = 1e-15;
        let phase = 0.0;

        // Beyond interaction range
        let distance = 200e-6; // Beyond 100 μm range
        let force = calc
            .secondary_bjerknes_force(r1, r2, v1, v2, phase, distance)
            .unwrap();

        assert_eq!(force.secondary, 0.0); // No force at large distance
        assert_eq!(force.interaction_type, InteractionType::Neutral);
    }

    #[test]
    fn test_invalid_radius() {
        let config = BjerknesConfig::default();
        let calc = BjerknesCalculator::new(config);

        let result = calc.primary_bjerknes_force(0.0, 100e3, 1e6);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_distance_error() {
        let config = BjerknesConfig::default();
        let calc = BjerknesCalculator::new(config);

        let result = calc.secondary_bjerknes_force(5e-6, 5e-6, 1e-15, 1e-15, 0.0, 0.0);
        assert!(result.is_err());
    }
}
