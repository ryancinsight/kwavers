//! Bubble-bubble interactions
//!
//! Calculates forces between bubbles (Bjerknes forces, etc.)

use super::bubble_state::BubbleState;
use ndarray::Array3;
use std::collections::HashMap;

/// Bubble interaction calculator
pub struct BubbleInteractions {
    /// Interaction cutoff distance [m]
    pub cutoff_distance: f64,
    /// Interaction strength factor
    pub interaction_strength: f64,
}

impl Default for BubbleInteractions {
    fn default() -> Self {
        Self {
            cutoff_distance: 1e-3, // 1 mm
            interaction_strength: 1.0,
        }
    }
}

impl BubbleInteractions {
    /// Calculate interaction pressure field from all bubbles
    pub fn calculate_interaction_field(
        &self,
        bubbles: &HashMap<(usize, usize, usize), BubbleState>,
        grid_shape: (usize, usize, usize),
        grid_spacing: (f64, f64, f64),
    ) -> Array3<f64> {
        let mut interaction_field = Array3::zeros(grid_shape);
        
        // For each grid point
        for i in 0..grid_shape.0 {
            for j in 0..grid_shape.1 {
                for k in 0..grid_shape.2 {
                    let mut total_interaction = 0.0;
                    
                    // Sum contributions from all bubbles
                    for ((bi, bj, bk), state) in bubbles {
                        if i == *bi && j == *bj && k == *bk {
                            continue; // Skip self-interaction
                        }
                        
                        // Calculate distance
                        let dx = (i as f64 - *bi as f64) * grid_spacing.0;
                        let dy = (j as f64 - *bj as f64) * grid_spacing.1;
                        let dz = (k as f64 - *bk as f64) * grid_spacing.2;
                        let distance = (dx*dx + dy*dy + dz*dz).sqrt();
                        
                        if distance < self.cutoff_distance && distance > 0.0 {
                            // Bjerknes force contribution
                            let bjerknes = self.calculate_bjerknes_contribution(
                                state,
                                distance,
                            );
                            total_interaction += bjerknes;
                        }
                    }
                    
                    interaction_field[[i, j, k]] = total_interaction * self.interaction_strength;
                }
            }
        }
        
        interaction_field
    }
    
    /// Calculate Bjerknes force contribution from a single bubble
    fn calculate_bjerknes_contribution(
        &self,
        bubble: &BubbleState,
        distance: f64,
    ) -> f64 {
        // Primary Bjerknes force: F ∝ V̇/r
        // Secondary Bjerknes force: F ∝ V₁V₂/r²
        
        let volume = bubble.volume();
        let volume_rate = 4.0 * std::f64::consts::PI * bubble.radius.powi(2) * bubble.wall_velocity;
        
        // Simplified model: pressure contribution decreases with distance
        let primary = volume_rate / (4.0 * std::f64::consts::PI * distance.powi(2));
        let secondary = volume * bubble.wall_acceleration / (distance.powi(2));
        
        primary + 0.1 * secondary
    }
}

/// Calculate Bjerknes force between two bubbles
pub struct BjerknesForce;

impl BjerknesForce {
    /// Primary Bjerknes force (bubble in pressure gradient)
    pub fn primary(
        bubble_volume: f64,
        pressure_gradient: f64,
    ) -> f64 {
        -bubble_volume * pressure_gradient
    }
    
    /// Secondary Bjerknes force (bubble-bubble interaction)
    pub fn secondary(
        bubble1: &BubbleState,
        bubble2: &BubbleState,
        distance: f64,
        liquid_density: f64,
    ) -> f64 {
        if distance <= 0.0 {
            return 0.0;
        }
        
        // Volume oscillation rates
        let v1_dot = 4.0 * std::f64::consts::PI * bubble1.radius.powi(2) * bubble1.wall_velocity;
        let v2_dot = 4.0 * std::f64::consts::PI * bubble2.radius.powi(2) * bubble2.wall_velocity;
        
        // Force magnitude
        let force = -liquid_density * v1_dot * v2_dot / (4.0 * std::f64::consts::PI * distance);
        
        force
    }
    
    /// Check if bubbles attract or repel
    pub fn interaction_type(bubble1: &BubbleState, bubble2: &BubbleState) -> InteractionType {
        // In phase: both expanding or both contracting -> attraction
        // Out of phase: one expanding, one contracting -> repulsion
        
        if bubble1.wall_velocity * bubble2.wall_velocity > 0.0 {
            InteractionType::Attraction
        } else {
            InteractionType::Repulsion
        }
    }
}

/// Type of bubble-bubble interaction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InteractionType {
    Attraction,
    Repulsion,
    Neutral,
}

/// Collective effects in bubble clouds
pub struct CollectiveEffects;

impl CollectiveEffects {
    /// Calculate effective sound speed in bubbly liquid
    pub fn wood_sound_speed(
        void_fraction: f64,
        liquid_density: f64,
        liquid_sound_speed: f64,
        gas_density: f64,
        gas_sound_speed: f64,
    ) -> f64 {
        // Wood's equation for sound speed in bubbly liquid
        let mixture_density = (1.0 - void_fraction) * liquid_density + void_fraction * gas_density;
        let compressibility = (1.0 - void_fraction) / (liquid_density * liquid_sound_speed.powi(2))
            + void_fraction / (gas_density * gas_sound_speed.powi(2));
        
        (mixture_density * compressibility).powf(-0.5)
    }
    
    /// Calculate void fraction from bubble field
    pub fn void_fraction(
        bubbles: &HashMap<(usize, usize, usize), BubbleState>,
        grid_volume: f64,
    ) -> f64 {
        let total_bubble_volume: f64 = bubbles.values()
            .map(|b| b.volume())
            .sum();
        
        total_bubble_volume / grid_volume
    }
    
    /// Estimate collective oscillation frequency
    pub fn collective_frequency(
        mean_radius: f64,
        void_fraction: f64,
        liquid_properties: (f64, f64), // (density, sound_speed)
    ) -> f64 {
        let (rho, c) = liquid_properties;
        
        // Minnaert frequency modified for void fraction
        let f0 = 1.0 / (2.0 * std::f64::consts::PI * mean_radius)
            * (3.0 * 1.4 * 101325.0 / rho).sqrt();
        
        // Correction for bubble-bubble interactions
        f0 * (1.0 - void_fraction).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
            BjerknesForce::interaction_type(&bubble1, &bubble2),
            InteractionType::Attraction
        );
        
        // Opposite phase
        bubble1.wall_velocity = 10.0;
        bubble2.wall_velocity = -5.0;
        assert_eq!(
            BjerknesForce::interaction_type(&bubble1, &bubble2),
            InteractionType::Repulsion
        );
    }
    
    #[test]
    fn test_wood_sound_speed() {
        let c_mixture = CollectiveEffects::wood_sound_speed(
            0.01,    // 1% void fraction
            1000.0,  // water density
            1500.0,  // water sound speed
            1.2,     // air density
            340.0,   // air sound speed
        );
        
        // Sound speed should be significantly reduced
        assert!(c_mixture < 1500.0);
        assert!(c_mixture > 100.0); // But still reasonable
    }
}