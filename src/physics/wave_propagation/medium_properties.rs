//! Medium properties for wave propagation calculations
//!
//! Defines acoustic and optical properties of propagation media.

use crate::constants::physics::{DENSITY_WATER, SOUND_SPEED_WATER};
use ndarray::Array2;

/// Medium properties for wave propagation
#[derive(Debug, Clone))]
pub struct MediumProperties {
    /// Wave speed [m/s]
    pub wave_speed: f64,
    /// Density [kg/m³]
    pub density: f64,
    /// Refractive index (optical)
    pub refractive_index: f64,
    /// Absorption coefficient [1/m]
    pub absorption: f64,
    /// Anisotropy tensor (for anisotropic media)
    pub anisotropy: Option<Array2<f64>>,
}

impl MediumProperties {
    /// Create properties for water at standard conditions
    pub fn water() -> Self {
        Self {
            wave_speed: SOUND_SPEED_WATER,
            density: DENSITY_WATER,
            refractive_index: 1.333, // At 20°C, 589 nm
            absorption: 0.0,
            anisotropy: None,
        }
    }

    /// Calculate acoustic impedance Z = ρc
    pub fn acoustic_impedance(&self) -> f64 {
        self.density * self.wave_speed
    }

    /// Calculate optical impedance Z = √(μ/ε) ≈ Z₀/n for non-magnetic media
    pub fn optical_impedance(&self) -> f64 {
        const VACUUM_IMPEDANCE: f64 = 376.730313668; // Ohms
        VACUUM_IMPEDANCE / self.refractive_index
    }

    /// Check if medium is anisotropic
    pub fn is_anisotropic(&self) -> bool {
        self.anisotropy.is_some()
    }

    /// Get wave speed along a specific direction (for anisotropic media)
    pub fn directional_wave_speed(&self, direction: &[f64; 3]) -> f64 {
        if let Some(ref tensor) = self.anisotropy {
            // Simplified calculation - full implementation would use Christoffel equation
            self.wave_speed
        } else {
            self.wave_speed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_water_properties() {
        let water = MediumProperties::water();
        assert_relative_eq!(water.wave_speed, SOUND_SPEED_WATER);
        assert_relative_eq!(water.density, DENSITY_WATER);
        assert_relative_eq!(water.refractive_index, 1.333);
    }

    #[test]
    fn test_acoustic_impedance() {
        let water = MediumProperties::water();
        let expected = DENSITY_WATER * SOUND_SPEED_WATER;
        assert_relative_eq!(water.acoustic_impedance(), expected);
    }
}
