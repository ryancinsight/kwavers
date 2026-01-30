//! Unified Material Properties (Consolidated from physics/materials)
//!
//! This module provides a unified `MaterialProperties` struct that combines all acoustic, thermal,
//! optical, and perfusion properties needed for multi-physics simulations.
//!
//! This replaces the `physics/materials/mod.rs` implementation, moving material property
//! definitions from the physics layer to the domain layer where they belong (layer fix).
//!
//! # Architecture
//!
//! Material properties are **domain specifications** (WHAT materials have), not physics equations
//! (HOW materials behave). Therefore, they belong in the domain layer.
//!
//! - **Domain Layer** (`domain/medium/properties/`): Material specifications ✅ CORRECT
//! - **Physics Layer**: Uses these properties in equations, doesn't define them ✅ CORRECT
//!
//! # Physical Foundation
//!
//! Properties are sourced from:
//! - Duck (1990) - Physical Properties of Tissues
//! - Perry & Green (2007) - Chemical Engineering Handbook
//! - IEC 61161:2013 - Ultrasound equipment safety
//! - FDA ultrasound safety guidelines
//!
//! # Examples
//!
//! ```rust,ignore
//! use kwavers::domain::medium::properties::{MaterialProperties, tissue, fluids, implants};
//!
//! // Tissue properties
//! let brain = tissue::BRAIN_WHITE_MATTER;
//! assert!(brain.validate().is_ok());
//!
//! // Fluid properties
//! let blood = fluids::WHOLE_BLOOD;
//! let r = brain.reflection_coefficient(&blood);
//!
//! // Implant properties
//! let titanium = implants::TITANIUM_GRADE5;
//! assert!(titanium.sound_speed > 5000.0);
//! ```

use crate::core::error::{KwaversError, KwaversResult};
use serde::{Deserialize, Serialize};

/// Unified material property structure
///
/// Contains all acoustic, thermal, optical, and perfusion properties needed for simulations.
/// This struct serves as the SSOT (Single Source of Truth) for material properties throughout
/// the kwavers library.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MaterialProperties {
    // ========================================================================
    // Acoustic Properties
    // ========================================================================
    /// Speed of sound [m/s]
    pub sound_speed: f64,

    /// Density [kg/m³]
    pub density: f64,

    /// Acoustic impedance Z = ρc [kg/(m²·s)]
    pub impedance: f64,

    /// Absorption coefficient [Np/m/MHz] (frequency-dependent: α = α₀·f^y)
    pub absorption_coefficient: f64,

    /// Absorption frequency exponent (typically 1.0-2.0)
    pub absorption_exponent: f64,

    /// Nonlinearity parameter B/A (Goldberg-Saffren)
    pub nonlinearity_parameter: f64,

    /// Shear viscosity [Pa·s]
    pub shear_viscosity: f64,

    /// Bulk viscosity [Pa·s]
    pub bulk_viscosity: f64,

    // ========================================================================
    // Thermal Properties
    // ========================================================================
    /// Specific heat capacity [J/(kg·K)]
    pub specific_heat: f64,

    /// Thermal conductivity [W/(m·K)]
    pub thermal_conductivity: f64,

    /// Thermal diffusivity α = k/(ρ·c) [m²/s]
    pub thermal_diffusivity: f64,

    // ========================================================================
    // Perfusion Properties (Tissue-Specific)
    // ========================================================================
    /// Blood perfusion rate [mL/100g/min]
    pub perfusion_rate: f64,

    /// Arterial blood temperature [°C]
    pub arterial_temperature: f64,

    /// Metabolic heat generation [W/kg]
    pub metabolic_heat: f64,

    // ========================================================================
    // Optical Properties
    // ========================================================================
    /// Absorption coefficient [1/m]
    pub optical_absorption: f64,

    /// Scattering coefficient [1/m]
    pub optical_scattering: f64,

    /// Refractive index [-]
    pub refractive_index: f64,

    // ========================================================================
    // State Reference
    // ========================================================================
    /// Temperature at which properties are defined [°C]
    pub reference_temperature: f64,

    /// Pressure at which properties are defined [Pa]
    pub reference_pressure: f64,
}

impl MaterialProperties {
    /// Create material properties with core parameters
    pub fn new(
        sound_speed: f64,
        density: f64,
        absorption_coefficient: f64,
        specific_heat: f64,
        thermal_conductivity: f64,
    ) -> Self {
        let impedance = density * sound_speed;
        let thermal_diffusivity = thermal_conductivity / (density * specific_heat);

        Self {
            sound_speed,
            density,
            impedance,
            absorption_coefficient,
            absorption_exponent: 1.0,
            nonlinearity_parameter: 0.0,
            shear_viscosity: 1e-3,
            bulk_viscosity: 1e-3,
            specific_heat,
            thermal_conductivity,
            thermal_diffusivity,
            perfusion_rate: 0.0,
            arterial_temperature: 37.0,
            metabolic_heat: 0.0,
            optical_absorption: 0.0,
            optical_scattering: 0.0,
            refractive_index: 1.0,
            reference_temperature: 20.0,
            reference_pressure: 101325.0,
        }
    }

    /// Validate material properties against physical constraints
    pub fn validate(&self) -> KwaversResult<()> {
        use crate::core::error::MediumError;

        // Speed of sound must be positive
        if self.sound_speed <= 0.0 {
            return Err(KwaversError::Medium(MediumError::InvalidProperties {
                property: "sound_speed".to_string(),
                value: self.sound_speed,
                constraint: "must be positive".to_string(),
            }));
        }

        // Density must be positive
        if self.density <= 0.0 {
            return Err(KwaversError::Medium(MediumError::InvalidProperties {
                property: "density".to_string(),
                value: self.density,
                constraint: "must be positive".to_string(),
            }));
        }

        // Impedance must be positive
        if self.impedance <= 0.0 {
            return Err(KwaversError::Medium(MediumError::InvalidProperties {
                property: "impedance".to_string(),
                value: self.impedance,
                constraint: "must be positive".to_string(),
            }));
        }

        // Absorption must be non-negative
        if self.absorption_coefficient < 0.0 {
            return Err(KwaversError::Medium(MediumError::InvalidProperties {
                property: "absorption_coefficient".to_string(),
                value: self.absorption_coefficient,
                constraint: "must be non-negative".to_string(),
            }));
        }

        // Specific heat must be positive
        if self.specific_heat <= 0.0 {
            return Err(KwaversError::Medium(MediumError::InvalidProperties {
                property: "specific_heat".to_string(),
                value: self.specific_heat,
                constraint: "must be positive".to_string(),
            }));
        }

        // Thermal conductivity must be non-negative
        if self.thermal_conductivity < 0.0 {
            return Err(KwaversError::Medium(MediumError::InvalidProperties {
                property: "thermal_conductivity".to_string(),
                value: self.thermal_conductivity,
                constraint: "must be non-negative".to_string(),
            }));
        }

        // Viscosity must be non-negative
        if self.shear_viscosity < 0.0 || self.bulk_viscosity < 0.0 {
            return Err(KwaversError::Medium(MediumError::InvalidProperties {
                property: "viscosity".to_string(),
                value: self.shear_viscosity.min(self.bulk_viscosity),
                constraint: "must be non-negative".to_string(),
            }));
        }

        // Refractive index should be typically >= 1.0 for physical materials
        if self.refractive_index < 1.0 {
            return Err(KwaversError::Medium(MediumError::InvalidProperties {
                property: "refractive_index".to_string(),
                value: self.refractive_index,
                constraint: "must be >= 1.0".to_string(),
            }));
        }

        Ok(())
    }

    /// Calculate acoustic impedance mismatch ratio
    pub fn impedance_ratio(&self, other: &MaterialProperties) -> f64 {
        let z1 = self.impedance;
        let z2 = other.impedance;
        ((z1 - z2) / (z1 + z2)).abs()
    }

    /// Calculate reflection coefficient at normal incidence
    pub fn reflection_coefficient(&self, other: &MaterialProperties) -> f64 {
        let z1 = self.impedance;
        let z2 = other.impedance;
        ((z2 - z1) / (z2 + z1)).abs()
    }

    /// Calculate transmission coefficient at normal incidence
    pub fn transmission_coefficient(&self, other: &MaterialProperties) -> f64 {
        1.0 - self.reflection_coefficient(other)
    }

    /// Absorption coefficient at given frequency
    pub fn absorption_at_frequency(&self, frequency: f64) -> f64 {
        self.absorption_coefficient * frequency.powf(self.absorption_exponent)
    }

    /// Attenuation in dB/cm at given frequency
    pub fn attenuation_db_cm(&self, frequency: f64) -> f64 {
        let alpha = self.absorption_at_frequency(frequency);
        // Convert Np/m to dB/cm: dB = 20·log₁₀(e) · Np · 0.01
        alpha * 20.0 * std::f64::consts::LOG10_E * 0.01
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_material_creation() {
        let mat = MaterialProperties::new(1500.0, 1000.0, 0.002, 4186.0, 0.6);
        assert_eq!(mat.sound_speed, 1500.0);
        assert_eq!(mat.impedance, 1500000.0);
    }

    #[test]
    fn test_validation_valid() {
        let mat = MaterialProperties::new(1500.0, 1000.0, 0.002, 4186.0, 0.6);
        assert!(mat.validate().is_ok());
    }

    #[test]
    fn test_validation_invalid_speed() {
        let mut mat = MaterialProperties::new(1500.0, 1000.0, 0.002, 4186.0, 0.6);
        mat.sound_speed = -1500.0;
        assert!(mat.validate().is_err());
    }

    #[test]
    fn test_impedance_match() {
        let mat = MaterialProperties::new(1500.0, 1000.0, 0.002, 4186.0, 0.6);
        let same = MaterialProperties::new(1500.0, 1000.0, 0.002, 4186.0, 0.6);
        assert!((mat.reflection_coefficient(&same)).abs() < 1e-6);
    }

    #[test]
    fn test_attenuation_frequency_dependence() {
        let mat = MaterialProperties::new(1500.0, 1000.0, 0.002, 4186.0, 0.6);
        let att_1mhz = mat.absorption_at_frequency(1e6);
        let att_2mhz = mat.absorption_at_frequency(2e6);
        // With exponent 1.0, doubling frequency doubles absorption
        assert!(att_2mhz > att_1mhz);
    }
}
