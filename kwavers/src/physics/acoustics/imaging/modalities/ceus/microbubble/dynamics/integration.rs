//! Velocity-Verlet integration for CEUS microbubble oscillation.

use super::BubbleDynamics;
use crate::core::error::KwaversResult;
use crate::domain::imaging::ultrasound::ceus::Microbubble;
use crate::physics::acoustics::imaging::modalities::ceus::microbubble::response::BubbleResponse;

impl BubbleDynamics {
    /// Simulate radial oscillation response to acoustic pressure.
    pub fn simulate_oscillation(
        &self,
        bubble: &Microbubble,
        acoustic_pressure: f64,
        frequency: f64,
        duration: f64,
    ) -> KwaversResult<BubbleResponse> {
        let n_steps = (duration / self.dt) as usize;
        let mut radius = vec![0.0; n_steps];
        let mut scattered_pressure = vec![0.0; n_steps];

        radius[0] = bubble.radius_eq;
        let mut radius_dot = 0.0;
        let r0 = bubble.radius_eq.max(1e-12);
        let p_gas0 = self.equilibrium_gas_pressure(bubble, r0);

        for i in 0..n_steps.saturating_sub(1) {
            let time = i as f64 * self.dt;
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

            let radius_new =
                radius[i] + radius_dot * self.dt + 0.5 * acceleration * self.dt * self.dt;
            let r_new = radius_new.max(1e-12);
            let radius_dot_pred = radius_dot + acceleration * self.dt;
            let time_new = (i + 1) as f64 * self.dt;
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

            radius_dot += 0.5 * (acceleration + acceleration_new) * self.dt;
            radius[i + 1] = radius_new;

            let volume_change =
                4.0 / 3.0 * std::f64::consts::PI * (radius_new.powi(3) - radius[i].powi(3));
            scattered_pressure[i + 1] = self.liquid_density * volume_change / self.dt;
        }

        Ok(BubbleResponse {
            time: (0..n_steps).map(|i| i as f64 * self.dt).collect(),
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
        let p_acoustic = acoustic_pressure * (2.0 * std::f64::consts::PI * frequency * time).sin();
        let p_gas = p_gas0 * (r0 / radius).powf(3.0 * bubble.polytropic_index);
        let p_surface = 2.0 * bubble.surface_tension / radius;
        let p_shell =
            4.0 * bubble.shell_elasticity * bubble.shell_thickness * (radius - r0) / (r0 * r0);
        let damping_force = -self.damping_coefficient * radius_dot / radius;
        let total_pressure = p_gas - p_surface - p_shell - self.ambient_pressure - p_acoustic;

        total_pressure / (self.liquid_density * radius) + damping_force
            - 1.5 * radius_dot * radius_dot / radius
    }
}
