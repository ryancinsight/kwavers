//! Analytical medium properties for wave propagation calculations
//!
//! This module provides structures and calculations for wave propagation
//! in different media, including acoustic and optical impedance calculations,
//! primarily for analytical solutions.

use crate::core::constants::{DENSITY_WATER, SOUND_SPEED_WATER};
use ndarray::Array2;

/// Medium properties for wave propagation
#[derive(Debug, Clone)]
pub struct AnalyticalMediumProperties {
    /// Wave speed in the medium [m/s]
    pub wave_speed: f64,
    /// Density for acoustic waves [kg/m³]
    pub density: f64,
    /// Refractive index for optical waves
    pub refractive_index: f64,
    /// Absorption coefficient [1/m]
    pub absorption: f64,
    /// Anisotropy tensor (for anisotropic media)
    pub anisotropy: Option<Array2<f64>>,
}

impl AnalyticalMediumProperties {
    /// Create properties for water at standard conditions
    #[must_use]
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
    #[must_use]
    pub fn acoustic_impedance(&self) -> f64 {
        self.density * self.wave_speed
    }

    /// Calculate optical impedance Z = √(μ/ε) ≈ Z₀/n for non-magnetic media
    #[must_use]
    pub fn optical_impedance(&self) -> f64 {
        const VACUUM_IMPEDANCE: f64 = 376.730313668; // Ohms
        VACUUM_IMPEDANCE / self.refractive_index
    }

    /// Generic impedance calculation (defaults to acoustic impedance)
    #[must_use]
    pub fn impedance(&self) -> f64 {
        self.acoustic_impedance()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_water_properties() {
        let water = AnalyticalMediumProperties::water();
        assert_relative_eq!(water.wave_speed, SOUND_SPEED_WATER, epsilon = 1e-10);
        assert_relative_eq!(water.density, DENSITY_WATER, epsilon = 1e-10);
        assert_relative_eq!(water.refractive_index, 1.333, epsilon = 1e-10);
    }

    #[test]
    fn test_acoustic_impedance() {
        let water = AnalyticalMediumProperties::water();
        let impedance = water.acoustic_impedance();
        let expected = DENSITY_WATER * SOUND_SPEED_WATER;
        assert_relative_eq!(impedance, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_optical_impedance() {
        let water = AnalyticalMediumProperties::water();
        let impedance = water.optical_impedance();
        let expected = 376.730313668 / 1.333;
        assert_relative_eq!(impedance, expected, epsilon = 1e-6);
    }
}
