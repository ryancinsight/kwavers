//! Snell's law implementation for wave refraction
//!
//! Implements Snell's law of refraction: n₁sin(θ₁) = n₂sin(θ₂)

use super::Interface;
use crate::error::{KwaversError, KwaversResult, PhysicsError};
use std::f64::consts::PI;

/// Critical angles for wave propagation
#[derive(Debug, Clone)]
pub struct CriticalAngles {
    /// Critical angle for total internal reflection [radians]
    pub total_internal_reflection: Option<f64>,
    /// Brewster's angle (polarization angle) [radians]
    pub brewster: Option<f64>,
}

/// Snell's law calculator
pub struct SnellLawCalculator<'a> {
    interface: &'a Interface,
    /// Ratio of wave speeds or refractive indices
    speed_ratio: f64,
}

impl<'a> SnellLawCalculator<'a> {
    /// Create a new Snell's law calculator
    pub fn new(interface: &'a Interface) -> Self {
        // For acoustic waves: use wave speeds
        // For optical waves: use refractive indices
        let speed_ratio = interface.medium1.wave_speed / interface.medium2.wave_speed;

        Self {
            interface,
            speed_ratio,
        }
    }

    /// Calculate transmitted angle using Snell's law
    pub fn calculate_transmitted_angle(&self, incident_angle: f64) -> KwaversResult<f64> {
        // Validate input
        if incident_angle < 0.0 || incident_angle > PI / 2.0 {
            return Err(KwaversError::Physics(PhysicsError::InvalidState {
                field: "incident_angle".to_string(),
                value: format!("{}", incident_angle),
                reason: "must be between 0 and π/2".to_string(),
            }));
        }

        // Apply Snell's law: n₁sin(θ₁) = n₂sin(θ₂)
        let n1 = self.interface.medium1.refractive_index;
        let n2 = self.interface.medium2.refractive_index;
        let sin_transmitted = (n1 / n2) * incident_angle.sin();

        // Check for total internal reflection
        if sin_transmitted > 1.0 {
            // Total internal reflection occurs
            // Return complex angle (evanescent wave)
            return Ok(PI / 2.0); // Grazing angle
        }

        Ok(sin_transmitted.asin())
    }

    /// Calculate critical angle for total internal reflection
    pub fn critical_angle(&self) -> Option<f64> {
        let n1 = self.interface.medium1.refractive_index;
        let n2 = self.interface.medium2.refractive_index;

        // Critical angle exists only when n1 > n2
        if n1 > n2 {
            Some((n2 / n1).asin())
        } else {
            None
        }
    }

    /// Calculate Brewster's angle (polarization angle)
    pub fn brewster_angle(&self) -> Option<f64> {
        // For optical waves: tan(θB) = n₂/n₁
        let n1 = self.interface.medium1.refractive_index;
        let n2 = self.interface.medium2.refractive_index;

        if n1 > 0.0 && n2 > 0.0 {
            Some((n2 / n1).atan())
        } else {
            None
        }
    }

    /// Calculate all critical angles
    pub fn critical_angles(&self) -> CriticalAngles {
        CriticalAngles {
            total_internal_reflection: self.critical_angle(),
            brewster: self.brewster_angle(),
        }
    }

    /// Check if total internal reflection occurs at given angle
    pub fn is_total_internal_reflection(&self, incident_angle: f64) -> bool {
        if let Some(critical) = self.critical_angle() {
            incident_angle > critical
        } else {
            false
        }
    }

    /// Calculate the evanescent wave decay constant for total internal reflection
    pub fn evanescent_decay_constant(&self, incident_angle: f64, wavelength: f64) -> Option<f64> {
        if !self.is_total_internal_reflection(incident_angle) {
            return None;
        }

        // Calculate decay constant κ = (2π/λ) * √(n₁²sin²θ - n₂²)
        let n1 = self.interface.medium1.refractive_index;
        let n2 = self.interface.medium2.refractive_index;
        let sin_i = incident_angle.sin();

        let discriminant = n1 * n1 * sin_i * sin_i - n2 * n2;
        if discriminant > 0.0 {
            Some((2.0 * PI / wavelength) * discriminant.sqrt())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::wave_propagation::{InterfaceType, MediumProperties};
    use crate::constants::optics::{SPEED_OF_LIGHT, WATER_REFRACTIVE_INDEX, GLASS_REFRACTIVE_INDEX};

    #[test]
    fn test_snells_law_water_to_glass() {
        let mut medium1 = MediumProperties::water();
        medium1.refractive_index = WATER_REFRACTIVE_INDEX;
        medium1.wave_speed = SPEED_OF_LIGHT / WATER_REFRACTIVE_INDEX;

        let mut medium2 = MediumProperties::water();
        medium2.refractive_index = GLASS_REFRACTIVE_INDEX;
        medium2.wave_speed = SPEED_OF_LIGHT / GLASS_REFRACTIVE_INDEX;

        let interface = Interface {
            medium1,
            medium2,
            normal: [0.0, 0.0, 1.0],
            position: [0.0, 0.0, 0.0],
            interface_type: InterfaceType::Planar,
        };

        let calc = SnellLawCalculator::new(&interface);

        // Test normal incidence
        let transmitted = calc.calculate_transmitted_angle(0.0).unwrap();
        assert!((transmitted - 0.0).abs() < 1e-10);

        // Test 30 degree incidence
        let incident = PI / 6.0;
        let transmitted = calc.calculate_transmitted_angle(incident).unwrap();

        // Verify Snell's law
        let n1_sin_i = WATER_REFRACTIVE_INDEX * incident.sin();
        let n2_sin_t = GLASS_REFRACTIVE_INDEX * transmitted.sin();
        assert!((n1_sin_i - n2_sin_t).abs() < 1e-10);
    }

    #[test]
    fn test_critical_angle() {
        let mut medium1 = MediumProperties::water();
        medium1.refractive_index = GLASS_REFRACTIVE_INDEX;
        medium1.wave_speed = SPEED_OF_LIGHT / GLASS_REFRACTIVE_INDEX;

        let mut medium2 = MediumProperties::water();
        medium2.refractive_index = 1.0; // Air
        medium2.wave_speed = 3e8;

        let interface = Interface {
            medium1,
            medium2,
            normal: [0.0, 0.0, 1.0],
            position: [0.0, 0.0, 0.0],
            interface_type: InterfaceType::Planar,
        };

        let calc = SnellLawCalculator::new(&interface);

        // Critical angle should be arcsin(1/1.5) ≈ 41.8°
        let critical = calc.critical_angle().unwrap();
        let expected = (1.0_f64 / 1.5).asin();
        assert!((critical - expected).abs() < 1e-10);

        // Test total internal reflection above critical angle
        assert!(calc.is_total_internal_reflection(critical + 0.1));
        assert!(!calc.is_total_internal_reflection(critical - 0.1));
    }

    #[test]
    fn test_brewster_angle() {
        let mut medium1 = MediumProperties::water();
        medium1.refractive_index = 1.0; // Air

        let mut medium2 = MediumProperties::water();
        medium2.refractive_index = 1.5; // Glass

        let interface = Interface {
            medium1,
            medium2,
            normal: [0.0, 0.0, 1.0],
            position: [0.0, 0.0, 0.0],
            interface_type: InterfaceType::Planar,
        };

        let calc = SnellLawCalculator::new(&interface);

        // Brewster's angle should be arctan(1.5/1.0) ≈ 56.3°
        let brewster = calc.brewster_angle().unwrap();
        let expected = 1.5_f64.atan();
        assert!((brewster - expected).abs() < 1e-10);
    }
}
