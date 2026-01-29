//! Tissue properties - Acoustic and thermal characteristics
//!
//! Provides standardized tissue property definitions based on:
//! - Duck (1990) - Physical Properties of Tissues and Equivalent Systems for Ultrasound and Elastography
//! - IEC 61161:2013 - Ultrasound equipment safety standard
//! - FDA guidance on ultrasound safety
//!
//! Temperature: 37°C (body temperature)
//! Pressure: 1 atm unless otherwise noted

use super::MaterialProperties;

/// Tissue material properties type alias
pub type TissueProperties = MaterialProperties;

// ============================================================================
// Reference Medium
// ============================================================================

/// Water (reference medium)
/// Temperature: 20°C (for comparison with literature)
pub const WATER: TissueProperties = TissueProperties {
    sound_speed: 1480.0,
    density: 998.2,
    impedance: 1480083.0,
    absorption_coefficient: 0.002,
    absorption_exponent: 2.0,
    nonlinearity_parameter: 5.0,
    shear_viscosity: 1.002e-3,
    bulk_viscosity: 0.0,
    specific_heat: 4186.0,
    thermal_conductivity: 0.6,
    thermal_diffusivity: 1.43e-7,
    perfusion_rate: 0.0,
    arterial_temperature: 20.0,
    metabolic_heat: 0.0,
    optical_absorption: 0.0,
    optical_scattering: 0.0,
    refractive_index: 1.33,
    reference_temperature: 20.0,
    reference_pressure: 101325.0,
};

// ============================================================================
// Brain Tissue (Transcranial Ultrasound)
// ============================================================================

/// Brain white matter (37°C)
/// Source: Duck (1990), Table 3.3
pub const BRAIN_WHITE_MATTER: TissueProperties = TissueProperties {
    sound_speed: 1540.0,
    density: 1040.0,
    impedance: 1601600.0,
    absorption_coefficient: 0.6,
    absorption_exponent: 1.0,
    nonlinearity_parameter: 6.5,
    shear_viscosity: 2e-3,
    bulk_viscosity: 5e-3,
    specific_heat: 3650.0,
    thermal_conductivity: 0.50,
    thermal_diffusivity: 1.33e-7,
    perfusion_rate: 75.0, // mL/100g/min
    arterial_temperature: 37.0,
    metabolic_heat: 0.7,       // W/kg
    optical_absorption: 2.0,   // 1/m (near-IR)
    optical_scattering: 100.0, // 1/m
    refractive_index: 1.37,
    reference_temperature: 37.0,
    reference_pressure: 101325.0,
};

/// Brain gray matter (37°C)
/// Source: Duck (1990), Table 3.3
pub const BRAIN_GRAY_MATTER: TissueProperties = TissueProperties {
    sound_speed: 1545.0,
    density: 1050.0,
    impedance: 1622250.0,
    absorption_coefficient: 0.7,
    absorption_exponent: 1.0,
    nonlinearity_parameter: 6.8,
    shear_viscosity: 2.2e-3,
    bulk_viscosity: 5.2e-3,
    specific_heat: 3680.0,
    thermal_conductivity: 0.52,
    thermal_diffusivity: 1.34e-7,
    perfusion_rate: 150.0, // mL/100g/min (higher than white matter)
    arterial_temperature: 37.0,
    metabolic_heat: 1.2, // W/kg (higher metabolic rate)
    optical_absorption: 2.5,
    optical_scattering: 110.0,
    refractive_index: 1.37,
    reference_temperature: 37.0,
    reference_pressure: 101325.0,
};

/// Skull (bone) - High impedance mismatch
/// Source: Duck (1990), Table 3.3
pub const SKULL: TissueProperties = TissueProperties {
    sound_speed: 4080.0,
    density: 1920.0,
    impedance: 7833600.0,
    absorption_coefficient: 3.0,
    absorption_exponent: 1.0,
    nonlinearity_parameter: 8.0,
    shear_viscosity: 5e-3,
    bulk_viscosity: 1e-2,
    specific_heat: 1300.0,
    thermal_conductivity: 0.4,
    thermal_diffusivity: 1.61e-7,
    perfusion_rate: 5.0, // Much lower perfusion in bone
    arterial_temperature: 37.0,
    metabolic_heat: 0.1, // Lower metabolic rate
    optical_absorption: 5.0,
    optical_scattering: 200.0,
    refractive_index: 1.65,
    reference_temperature: 37.0,
    reference_pressure: 101325.0,
};

// ============================================================================
// Liver Tissue
// ============================================================================

/// Liver (hepatic tissue) (37°C)
/// Source: Duck (1990), Table 3.3
pub const LIVER: TissueProperties = TissueProperties {
    sound_speed: 1570.0,
    density: 1070.0,
    impedance: 1679900.0,
    absorption_coefficient: 0.4,
    absorption_exponent: 1.0,
    nonlinearity_parameter: 7.0,
    shear_viscosity: 2e-3,
    bulk_viscosity: 5e-3,
    specific_heat: 3590.0,
    thermal_conductivity: 0.56,
    thermal_diffusivity: 1.46e-7,
    perfusion_rate: 100.0, // mL/100g/min
    arterial_temperature: 37.0,
    metabolic_heat: 0.8, // W/kg
    optical_absorption: 3.0,
    optical_scattering: 120.0,
    refractive_index: 1.38,
    reference_temperature: 37.0,
    reference_pressure: 101325.0,
};

// ============================================================================
// Kidney Tissue
// ============================================================================

/// Kidney cortex (37°C)
/// Source: Duck (1990), Table 3.3
pub const KIDNEY_CORTEX: TissueProperties = TissueProperties {
    sound_speed: 1560.0,
    density: 1050.0,
    impedance: 1638000.0,
    absorption_coefficient: 0.5,
    absorption_exponent: 1.0,
    nonlinearity_parameter: 6.8,
    shear_viscosity: 2e-3,
    bulk_viscosity: 5e-3,
    specific_heat: 3600.0,
    thermal_conductivity: 0.50,
    thermal_diffusivity: 1.39e-7,
    perfusion_rate: 120.0, // Very high perfusion
    arterial_temperature: 37.0,
    metabolic_heat: 1.0,
    optical_absorption: 2.5,
    optical_scattering: 110.0,
    refractive_index: 1.37,
    reference_temperature: 37.0,
    reference_pressure: 101325.0,
};

/// Kidney medulla (37°C)
pub const KIDNEY_MEDULLA: TissueProperties = TissueProperties {
    sound_speed: 1565.0,
    density: 1055.0,
    impedance: 1651575.0,
    absorption_coefficient: 0.5,
    absorption_exponent: 1.0,
    nonlinearity_parameter: 6.8,
    shear_viscosity: 2e-3,
    bulk_viscosity: 5e-3,
    specific_heat: 3600.0,
    thermal_conductivity: 0.50,
    thermal_diffusivity: 1.39e-7,
    perfusion_rate: 130.0,
    arterial_temperature: 37.0,
    metabolic_heat: 1.1,
    optical_absorption: 2.5,
    optical_scattering: 110.0,
    refractive_index: 1.37,
    reference_temperature: 37.0,
    reference_pressure: 101325.0,
};

// ============================================================================
// Other Common Tissues
// ============================================================================

/// Blood (37°C)
pub const BLOOD: TissueProperties = TissueProperties {
    sound_speed: 1584.0,
    density: 1060.0,
    impedance: 1679040.0,
    absorption_coefficient: 0.15,
    absorption_exponent: 1.0,
    nonlinearity_parameter: 5.5,
    shear_viscosity: 4e-3,
    bulk_viscosity: 0.0,
    specific_heat: 3650.0,
    thermal_conductivity: 0.54,
    thermal_diffusivity: 1.39e-7,
    perfusion_rate: 0.0, // Blood IS the perfusion medium
    arterial_temperature: 37.0,
    metabolic_heat: 0.0,
    optical_absorption: 100.0, // Blood strongly absorbs light
    optical_scattering: 500.0, // High scattering
    refractive_index: 1.335,
    reference_temperature: 37.0,
    reference_pressure: 101325.0,
};

/// Muscle (37°C)
pub const MUSCLE: TissueProperties = TissueProperties {
    sound_speed: 1580.0,
    density: 1070.0,
    impedance: 1690600.0,
    absorption_coefficient: 0.13,
    absorption_exponent: 1.0,
    nonlinearity_parameter: 7.0,
    shear_viscosity: 2e-3,
    bulk_viscosity: 5e-3,
    specific_heat: 3750.0,
    thermal_conductivity: 0.60,
    thermal_diffusivity: 1.48e-7,
    perfusion_rate: 80.0,
    arterial_temperature: 37.0,
    metabolic_heat: 1.2, // Active metabolism
    optical_absorption: 0.1,
    optical_scattering: 50.0,
    refractive_index: 1.38,
    reference_temperature: 37.0,
    reference_pressure: 101325.0,
};

/// Fat (adipose tissue) (37°C)
pub const FAT: TissueProperties = TissueProperties {
    sound_speed: 1450.0,  // Slower than water
    density: 900.0,       // Less dense
    impedance: 1305000.0, // Lower impedance
    absorption_coefficient: 0.48,
    absorption_exponent: 1.0,
    nonlinearity_parameter: 6.0,
    shear_viscosity: 2e-3,
    bulk_viscosity: 5e-3,
    specific_heat: 2500.0,        // Lower heat capacity
    thermal_conductivity: 0.20,   // Much lower thermal conductivity
    thermal_diffusivity: 8.89e-8, // Slower thermal diffusion
    perfusion_rate: 30.0,         // Lower perfusion
    arterial_temperature: 37.0,
    metabolic_heat: 0.2,      // Lower metabolic rate
    optical_absorption: 0.01, // Transparent
    optical_scattering: 10.0,
    refractive_index: 1.46,
    reference_temperature: 37.0,
    reference_pressure: 101325.0,
};

/// Cerebrospinal fluid (CSF) (37°C)
pub const CSF: TissueProperties = TissueProperties {
    sound_speed: 1515.0,
    density: 1007.0,
    impedance: 1524605.0,
    absorption_coefficient: 0.0,
    absorption_exponent: 1.0,
    nonlinearity_parameter: 5.0,
    shear_viscosity: 0.7e-3,
    bulk_viscosity: 0.0,
    specific_heat: 3900.0,
    thermal_conductivity: 0.60,
    thermal_diffusivity: 1.53e-7,
    perfusion_rate: 0.0, // Not perfused tissue
    arterial_temperature: 37.0,
    metabolic_heat: 0.0,
    optical_absorption: 0.0,
    optical_scattering: 0.0,
    refractive_index: 1.334,
    reference_temperature: 37.0,
    reference_pressure: 101325.0,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tissue_properties_valid() {
        assert!(WATER.validate().is_ok());
        assert!(BRAIN_WHITE_MATTER.validate().is_ok());
        assert!(SKULL.validate().is_ok());
        assert!(LIVER.validate().is_ok());
        assert!(BLOOD.validate().is_ok());
    }

    #[test]
    fn test_impedance_calculation() {
        let expected_water = 998.2 * 1480.0;
        assert!((WATER.impedance - expected_water).abs() < 1.0);
    }

    #[test]
    fn test_water_skull_reflection() {
        // Water-skull has high impedance mismatch
        let r = WATER.reflection_coefficient(&SKULL);
        assert!(r > 0.5); // More than 50% reflection
    }

    #[test]
    fn test_brain_tissue_difference() {
        // Gray matter should have different properties than white matter
        assert_ne!(
            BRAIN_GRAY_MATTER.sound_speed,
            BRAIN_WHITE_MATTER.sound_speed
        );
        assert!(BRAIN_GRAY_MATTER.perfusion_rate > BRAIN_WHITE_MATTER.perfusion_rate);
    }

    #[test]
    fn test_attenuation_at_frequency() {
        let att_1mhz = BRAIN_WHITE_MATTER.absorption_at_frequency(1e6);
        let att_3mhz = BRAIN_WHITE_MATTER.absorption_at_frequency(3e6);
        assert!(att_3mhz > att_1mhz);
    }
}
