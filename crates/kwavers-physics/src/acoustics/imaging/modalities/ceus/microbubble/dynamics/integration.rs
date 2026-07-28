//! Velocity-Verlet integration for CEUS microbubble oscillation.

use super::BubbleDynamics;
use crate::acoustics::imaging::modalities::ceus::microbubble::response::BubbleResponse;
use aequitas::systems::si::quantities::{Frequency, Pressure, Time};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::KwaversResult;
use kwavers_imaging::ultrasound::ceus::Microbubble;

impl BubbleDynamics {
    /// Simulate radial oscillation response to acoustic pressure.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn simulate_oscillation(
        &self,
        bubble: &Microbubble,
        acoustic_pressure: Pressure<f64>,
        frequency: Frequency<f64>,
        duration: Time<f64>,
    ) -> KwaversResult<BubbleResponse> {
        let acoustic_pressure = acoustic_pressure.into_base();
        let frequency = frequency.into_base();
        let duration = duration.into_base();
        let dt = self.dt.into_base();
        let liquid_density = self.liquid_density.into_base();
        let n_steps = (duration / dt) as usize;
        let mut radius = vec![0.0; n_steps];
        let mut scattered_pressure = vec![0.0; n_steps];

        radius[0] = bubble.radius_eq.into_base();
        let mut radius_dot = 0.0;
        let r0 = bubble.radius_eq.into_base().max(1e-12);
        let p_gas0 = self.equilibrium_gas_pressure(bubble, r0);

        for i in 0..n_steps.saturating_sub(1) {
            let time = i as f64 * dt;
            let r = radius[i].max(1e-12);
            let acceleration = self.wall_acceleration(
                bubble,
                acoustic_pressure,
                frequency,
                time,
                r,
                radius_dot,
                p_gas0,
                r0,
            );

            let radius_new = (0.5 * acceleration * dt).mul_add(dt, radius[i] + radius_dot * dt);
            let r_new = radius_new.max(1e-12);
            let radius_dot_pred = radius_dot + acceleration * dt;
            let time_new = (i + 1) as f64 * dt;
            let acceleration_new = self.wall_acceleration(
                bubble,
                acoustic_pressure,
                frequency,
                time_new,
                r_new,
                radius_dot_pred,
                p_gas0,
                r0,
            );

            radius_dot += 0.5 * (acceleration + acceleration_new) * dt;
            radius[i + 1] = radius_new;

            // Far-field scattered pressure from a pulsating spherical source
            // at a 1-m reference distance (Leighton 1994 Eq. 4.18):
            //   p_scat(r=1, t) = (ρ / r) · (R²·R̈ + 2·R·Ṙ²)
            // Prior to the 2026-05-21 fix this stored ρ·dV/dt (units kg/s,
            // not Pa) — a dimensional error that mis-labelled the volume
            // mass-flow rate as a pressure.
            let r_w = r_new;
            let v_w = radius_dot;
            let a_w = acceleration_new;
            scattered_pressure[i + 1] = liquid_density * (r_w * r_w * a_w + 2.0 * r_w * v_w * v_w);
        }

        Ok(BubbleResponse {
            time: (0..n_steps).map(|i| i as f64 * dt).collect(),
            radius,
            scattered_pressure,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn wall_acceleration(
        &self,
        bubble: &Microbubble,
        acoustic_pressure: f64,
        frequency: f64,
        time: f64,
        radius: f64,
        radius_dot: f64,
        p_gas0: f64,
        r0: f64,
    ) -> f64 {
        let p_acoustic = acoustic_pressure * (TWO_PI * frequency * time).sin();
        let p_gas = p_gas0 * (r0 / radius).powf(3.0 * bubble.polytropic_index);
        let p_surface = 2.0 * bubble.surface_tension.into_base() / radius;
        let p_shell = 4.0
            * bubble.shell_elasticity.into_base()
            * bubble.shell_thickness.into_base()
            * (radius - r0)
            / (r0 * r0);
        let damping_force = -self.damping_coefficient * radius_dot / radius;
        let total_pressure =
            p_gas - p_surface - p_shell - self.ambient_pressure.into_base() - p_acoustic;

        total_pressure / (self.liquid_density.into_base() * radius) + damping_force
            - 1.5 * radius_dot * radius_dot / radius
    }
}
