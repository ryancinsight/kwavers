//! Kinematics and coalescence prediction under Bjerknes forces

use super::calculator::BjerknesCalculator;
use kwavers_core::error::{KwaversError, KwaversResult};

impl BjerknesCalculator {
    /// Predict bubble trajectory under Bjerknes forces.
    ///
    /// ## Effective inertial mass
    ///
    /// For a gas bubble translating through an incompressible inviscid liquid,
    /// the gas-phase mass is negligible (`ρ_gas ≪ ρ_liquid`) and the inertia
    /// is dominated by the *added mass* of the displaced liquid.  For a sphere
    /// the added-mass coefficient is `C_a = 1/2`, giving
    ///
    /// ```text
    /// m_eff ≈ C_a · ρ_liquid · V = (1/2) · ρ · (4π/3) R³
    /// ```
    ///
    /// Reference: Lamb H. (1932) *Hydrodynamics*, 6th ed., §92 (added mass of a
    /// translating sphere); Crum LA (1975) JASA 57(6), 1363–1370 (Bjerknes-force
    /// translational dynamics of microbubbles).  The previous implementation used
    /// `3·ρ·V` instead of `½·ρ·V`, under-predicting bubble acceleration by 6×.
    ///
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if `bubble_radius` ≤ 0 or `time_step` ≤ 0.
    pub fn predict_bubble_motion(
        &self,
        bubble_radius: f64,
        bjerknes_force: f64,
        initial_velocity: f64,
        time_step: f64,
    ) -> KwaversResult<(f64, f64)> {
        if bubble_radius <= 0.0 || time_step <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Bubble radius and time step must be positive".to_owned(),
            ));
        }

        // Added-mass effective inertia of a translating sphere in inviscid fluid.
        let bubble_volume = (4.0 / 3.0) * std::f64::consts::PI * bubble_radius.powi(3);
        let effective_mass = 0.5 * self.config.rho * bubble_volume;

        let acceleration = bjerknes_force / effective_mass;

        // Update velocity and position (simple Euler integration)
        let new_velocity = initial_velocity + acceleration * time_step;
        let displacement =
            initial_velocity.mul_add(time_step, 0.5 * acceleration * time_step.powi(2));

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

#[cfg(test)]
mod tests {
    use super::super::types::BjerknesConfig;
    use super::*;

    fn calc() -> BjerknesCalculator {
        BjerknesCalculator::new(BjerknesConfig::default())
    }

    /// predict_bubble_motion: zero initial velocity → displacement = 0.5·a·dt²
    /// with the *added-mass* effective inertia (Lamb 1932 §92).
    ///
    /// With F=1e-9 N, R=1e-6 m (1 µm), ρ=1000 kg/m³:
    ///   V = (4/3)π·R³; m_eff = (1/2)·ρ·V; a = F/m_eff; d = 0.5·a·dt²
    #[test]
    fn predict_bubble_motion_zero_initial_velocity_matches_added_mass_kinematics() {
        let bjerknes_force = 1e-9_f64; // 1 nN
        let radius = 1e-6_f64; // 1 µm
        let dt = 1e-7_f64; // 0.1 µs

        use std::f64::consts::PI;
        let rho = calc().config.rho;
        let volume = (4.0 / 3.0) * PI * radius.powi(3);
        let effective_mass = 0.5 * rho * volume; // added mass C_a = 1/2
        let acceleration = bjerknes_force / effective_mass;
        let expected_displacement = 0.5 * acceleration * dt.powi(2);
        let expected_velocity = acceleration * dt;

        let (disp, vel) = calc()
            .predict_bubble_motion(radius, bjerknes_force, 0.0, dt)
            .unwrap();

        assert!(
            (disp - expected_displacement).abs() < 1e-30,
            "displacement: expected {expected_displacement:.4e}, got {disp:.4e}"
        );
        assert!(
            (vel - expected_velocity).abs() < 1e-30,
            "velocity: expected {expected_velocity:.4e}, got {vel:.4e}"
        );
    }

    /// predict_bubble_motion rejects zero/negative radius and time step.
    #[test]
    fn predict_bubble_motion_rejects_invalid_inputs() {
        let c = calc();
        assert!(
            c.predict_bubble_motion(0.0, 1e-9, 0.0, 1e-7).is_err(),
            "zero radius must be rejected"
        );
        assert!(
            c.predict_bubble_motion(1e-6, 1e-9, 0.0, 0.0).is_err(),
            "zero dt must be rejected"
        );
    }

    /// coalescence_probability is zero when distance exceeds threshold.
    #[test]
    fn coalescence_probability_zero_when_too_far() {
        let c = calc();
        let far = c.config.coalescence_distance * 2.0;
        assert_eq!(
            c.coalescence_probability(far, 1.0),
            0.0,
            "probability must be zero at distance > threshold"
        );
    }

    /// coalescence_probability is in [0, 1] for all physical inputs.
    #[test]
    fn coalescence_probability_in_unit_interval() {
        let c = calc();
        let threshold = c.config.coalescence_distance;
        for &d in &[0.0, threshold * 0.25, threshold * 0.5, threshold * 0.99] {
            for &v in &[0.0, 0.5, 1.0, 2.0] {
                let p = c.coalescence_probability(d, v);
                assert!(
                    (0.0..=1.0).contains(&p),
                    "probability {p} out of [0,1] at d={d:.2e}, v={v}"
                );
            }
        }
    }
}
