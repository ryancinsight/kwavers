//! Kinematics and coalescence prediction under Bjerknes forces

use super::calculator::BjerknesCalculator;
use crate::core::error::{KwaversError, KwaversResult};

impl BjerknesCalculator {
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
        let bubble_volume = (4.0 / 3.0) * std::f64::consts::PI * bubble_radius.powi(3);
        let effective_mass = 3.0 * self.config.rho * bubble_volume; // With added mass factor

        // Calculate acceleration
        let acceleration = bjerknes_force / effective_mass;

        // Update velocity and position (simple Euler integration)
        let new_velocity = initial_velocity + acceleration * time_step;
        let displacement = initial_velocity * time_step + 0.5 * acceleration * time_step.powi(2);

        Ok((displacement, new_velocity))
    }

    /// Calculate coalescence probability based on approach distance
    #[must_use]
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
}
