//! Bubble-bubble interactions: coalescence, radiation forces, dissolution
//!
//! ## Secondary Radiation Force (Bjerknes)
//!
//! ```text
//! F_rad = (4π/3) R₀³ ρ (2πf)² (p₀²/(3ρc²)) · V_amp · k
//! ```
//!
//! ## Coalescence
//!
//! Conservation of gas volume: V_new = V₁ + V₂
//! Conservation of momentum: m_new v_new = m₁ v₁ + m₂ v₂
//!
//! ## References
//!
//! - Bjerknes (1906), acoustic streaming and radiation pressure
//! - King (1934), radiation force on spheres

use super::config::CloudBubble;
use super::simulator::CloudDynamics;
use crate::core::error::KwaversResult;
use crate::physics::acoustics::imaging::modalities::ceus::microbubble::BubbleResponse;

impl CloudDynamics {
    /// Handle bubble-bubble interactions (coalescence check)
    pub(crate) fn handle_interactions(&mut self) -> KwaversResult<()> {
        let mut coalescence_events = Vec::new();

        for i in 0..self.bubbles.len() {
            if !self.bubbles[i].active {
                continue;
            }

            for j in (i + 1)..self.bubbles.len() {
                if !self.bubbles[j].active {
                    continue;
                }

                let dist = self.bubble_distance(i, j);
                if dist < self.config.coalescence_distance {
                    coalescence_events.push((i, j));
                }
            }
        }

        for (i, j) in coalescence_events {
            self.coalesce_bubbles(i, j);
        }

        Ok(())
    }

    /// Calculate distance between bubbles
    fn bubble_distance(&self, i: usize, j: usize) -> f64 {
        let dx = self.bubbles[i].position[0] - self.bubbles[j].position[0];
        let dy = self.bubbles[i].position[1] - self.bubbles[j].position[1];
        let dz = self.bubbles[i].position[2] - self.bubbles[j].position[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Handle bubble coalescence with volume and momentum conservation
    fn coalesce_bubbles(&mut self, i: usize, j: usize) {
        let bubble1 = &self.bubbles[i];
        let bubble2 = &self.bubbles[j];

        // Conservation of volume (spherical bubbles)
        let vol1 = (4.0 / 3.0) * std::f64::consts::PI * bubble1.current_radius.powi(3);
        let vol2 = (4.0 / 3.0) * std::f64::consts::PI * bubble2.current_radius.powi(3);
        let total_vol = vol1 + vol2;

        let new_radius = ((3.0 * total_vol) / (4.0 * std::f64::consts::PI)).cbrt();

        // Conservation of momentum
        let mass1 = vol1 * 1000.0;
        let mass2 = vol2 * 1000.0;
        let total_mass = mass1 + mass2;

        let mut new_velocity = [0.0; 3];
        for (k, val) in new_velocity.iter_mut().enumerate() {
            *val = (bubble1.velocity[k] * mass1 + bubble2.velocity[k] * mass2) / total_mass;
        }

        // New position (center of mass)
        let mut new_position = [0.0; 3];
        for (k, val) in new_position.iter_mut().enumerate() {
            *val = (bubble1.position[k] * mass1 + bubble2.position[k] * mass2) / total_mass;
        }

        // Update bubble i to be the merged bubble
        self.bubbles[i].current_radius = new_radius;
        self.bubbles[i].velocity = new_velocity;
        self.bubbles[i].position = new_position;
        self.bubbles[i].properties.radius_eq = new_radius;

        // Deactivate bubble j
        self.bubbles[j].active = false;
    }

    /// Handle bubble dissolution
    pub(crate) fn handle_dissolution(&mut self) -> KwaversResult<()> {
        for bubble in &mut self.bubbles {
            if bubble.active && bubble.current_radius < 0.5e-6 {
                bubble.active = false;
            }
        }
        Ok(())
    }

    /// Calculate primary radiation force on a bubble (Bjerknes force)
    pub(crate) fn calculate_radiation_force(
        &self,
        bubble: &CloudBubble,
        response: &BubbleResponse,
    ) -> [f64; 3] {
        if let Some(field) = &self.incident_field {
            let omega = 2.0 * std::f64::consts::PI * field.frequency;
            let k = omega / field.sound_speed;

            // Volume oscillation amplitude from radial pulsation
            let volume_amp = if response.radius.len() > 1 {
                let r0 = response.radius[0];
                let r_max = response.radius.iter().cloned().fold(0.0_f64, f64::max);
                (r_max - r0) / r0
            } else {
                0.0
            };

            // Radiation force magnitude
            let force_magnitude = (4.0 / 3.0)
                * std::f64::consts::PI
                * bubble.current_radius.powi(3)
                * field.pressure_amplitude
                * volume_amp
                * k;

            [force_magnitude, 0.0, 0.0]
        } else {
            [0.0, 0.0, 0.0]
        }
    }
}
