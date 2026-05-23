//! Tissue property catalogs
//!
//! Provides standardized tissue property definitions based on:
//! - Duck (1990) - Physical Properties of Tissues and Equivalent Systems for Ultrasound and Elastography
//! - IEC 61161:2013 - Ultrasound equipment safety standard
//! - FDA guidance on ultrasound safety
//!
//! Temperature: 37°C (body temperature)
//! Pressure: 1 atm unless otherwise noted

use super::material::AcousticMaterialProperties;
use crate::core::constants::acoustic_parameters::{DENSITY_SKULL, VISCOSITY_PARENCHYMAL_TISSUE};
use crate::core::constants::optical::{
    REFRACTIVE_INDEX_BIOLOGICAL_FLUID, REFRACTIVE_INDEX_BRAIN_TISSUE,
    REFRACTIVE_INDEX_SOFT_TISSUE_NIR, REFRACTIVE_INDEX_WATER,
};
use crate::core::constants::cavitation::VISCOSITY_WATER;
use crate::core::constants::fundamental::{
    ATMOSPHERIC_PRESSURE, B_OVER_A_BLOOD, B_OVER_A_BONE, B_OVER_A_BRAIN, B_OVER_A_CSF,
    B_OVER_A_FAT, B_OVER_A_KIDNEY, B_OVER_A_LIVER, B_OVER_A_MUSCLE, B_OVER_A_WATER, DENSITY_BLOOD,
    DENSITY_BRAIN, DENSITY_FAT, DENSITY_LIVER, DENSITY_MUSCLE, DENSITY_TISSUE, DENSITY_WATER,
    SOUND_SPEED_BLOOD, SOUND_SPEED_BRAIN, SOUND_SPEED_FAT, SOUND_SPEED_KIDNEY, SOUND_SPEED_LIVER,
    SOUND_SPEED_MUSCLE, SOUND_SPEED_WATER,
};
use crate::core::constants::thermodynamic::{
    BODY_TEMPERATURE_C, ROOM_TEMPERATURE_C, SPECIFIC_HEAT_BLOOD, SPECIFIC_HEAT_BONE,
    SPECIFIC_HEAT_BRAIN_GRAY, SPECIFIC_HEAT_BRAIN_WHITE, SPECIFIC_HEAT_CSF, SPECIFIC_HEAT_FAT,
    SPECIFIC_HEAT_LIVER, SPECIFIC_HEAT_MUSCLE, SPECIFIC_HEAT_TISSUE, SPECIFIC_HEAT_WATER,
    THERMAL_CONDUCTIVITY_BLOOD, THERMAL_CONDUCTIVITY_BRAIN, THERMAL_CONDUCTIVITY_BRAIN_GRAY,
    THERMAL_CONDUCTIVITY_CSF, THERMAL_CONDUCTIVITY_FAT, THERMAL_CONDUCTIVITY_KIDNEY,
    THERMAL_CONDUCTIVITY_LIVER, THERMAL_CONDUCTIVITY_MUSCLE, THERMAL_CONDUCTIVITY_SKULL,
    THERMAL_CONDUCTIVITY_WATER, THERMAL_DIFFUSIVITY_WATER,
};

/// Tissue material properties type alias
pub type TissueProperties = AcousticMaterialProperties;

// ============================================================================
// Reference Medium
// ============================================================================

/// Water (reference medium)
/// Temperature: 20°C (for comparison with literature)
pub const WATER: TissueProperties = TissueProperties {
    sound_speed: SOUND_SPEED_WATER,
    density: DENSITY_WATER,
    // Z = ρ·c = DENSITY_WATER × SOUND_SPEED_WATER = 998.2 × 1482.0 = 1 479 332.4 Pa·s/m
    // Derived from canonical constants (SSOT); do not hardcode.
    impedance: DENSITY_WATER * SOUND_SPEED_WATER,
    absorption_coefficient: 0.002,
    absorption_exponent: 2.0,
    nonlinearity_parameter: B_OVER_A_WATER, // 5.2 at 20°C (Duck 1990 Table 4.16)
    shear_viscosity: VISCOSITY_WATER,
    bulk_viscosity: 0.0,
    specific_heat: SPECIFIC_HEAT_WATER,
    thermal_conductivity: THERMAL_CONDUCTIVITY_WATER,
    thermal_diffusivity: THERMAL_DIFFUSIVITY_WATER,
    perfusion_rate: 0.0,
    arterial_temperature: ROOM_TEMPERATURE_C,
    metabolic_heat: 0.0,
    optical_absorption: 0.0,
    optical_scattering: 0.0,
    refractive_index: REFRACTIVE_INDEX_WATER,
    reference_temperature: ROOM_TEMPERATURE_C,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

// ============================================================================
// Brain Tissue (Transcranial Ultrasound)
// ============================================================================

/// Brain white matter (37°C)
/// Source: Duck (1990), Table 3.3
/// Z = ρ·c = 1040 × 1546 = 1 607 840 Pa·s/m
pub const BRAIN_WHITE_MATTER: TissueProperties = TissueProperties {
    sound_speed: SOUND_SPEED_BRAIN,
    density: DENSITY_BRAIN,
    impedance: 1_607_840.0,
    absorption_coefficient: 0.6,
    absorption_exponent: 1.0,
    nonlinearity_parameter: B_OVER_A_BRAIN, // 6.55 (Duck 1990 Table 4.16)
    shear_viscosity: VISCOSITY_PARENCHYMAL_TISSUE,
    bulk_viscosity: 5e-3,
    specific_heat: SPECIFIC_HEAT_BRAIN_WHITE,
    thermal_conductivity: THERMAL_CONDUCTIVITY_BRAIN,
    // α = k/(ρ·cp) = 0.50 / (1040 × 3650) = 1.317e-7 m²/s
    thermal_diffusivity: 1.317e-7,
    perfusion_rate: 75.0, // mL/100g/min
    arterial_temperature: BODY_TEMPERATURE_C,
    metabolic_heat: 0.7,       // W/kg
    optical_absorption: 2.0,   // 1/m (near-IR)
    optical_scattering: 100.0, // 1/m
    refractive_index: REFRACTIVE_INDEX_BRAIN_TISSUE,
    reference_temperature: BODY_TEMPERATURE_C,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

/// Brain gray matter (37°C)
/// Source: Duck (1990), Table 3.3
pub const BRAIN_GRAY_MATTER: TissueProperties = TissueProperties {
    sound_speed: 1545.0,
    density: DENSITY_TISSUE,
    impedance: 1622250.0,
    absorption_coefficient: 0.7,
    absorption_exponent: 1.0,
    nonlinearity_parameter: B_OVER_A_BRAIN, // 6.55 (Duck 1990 Table 4.16 brain mean)
    shear_viscosity: 2.2e-3,
    bulk_viscosity: 5.2e-3,
    specific_heat: SPECIFIC_HEAT_BRAIN_GRAY,
    thermal_conductivity: THERMAL_CONDUCTIVITY_BRAIN_GRAY,
    // α = k/(ρ·cp) = 0.52 / (1050 × 3680) = 1.346e-7 m²/s
    thermal_diffusivity: 1.346e-7,
    perfusion_rate: 150.0, // mL/100g/min (higher than white matter)
    arterial_temperature: BODY_TEMPERATURE_C,
    metabolic_heat: 1.2, // W/kg (higher metabolic rate)
    optical_absorption: 2.5,
    optical_scattering: 110.0,
    refractive_index: REFRACTIVE_INDEX_BRAIN_TISSUE,
    reference_temperature: BODY_TEMPERATURE_C,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

/// Skull (bone) - High impedance mismatch
/// Source: Duck (1990), Table 3.3
pub const SKULL: TissueProperties = TissueProperties {
    sound_speed: 4080.0,
    density: DENSITY_SKULL,
    // Z = ρ·c = 1920 × 4080 = 7 833 600 Pa·s/m
    impedance: 7833600.0,
    absorption_coefficient: 3.0,
    absorption_exponent: 1.0,
    nonlinearity_parameter: B_OVER_A_BONE, // 8.0 (Duck 1990 Table 4.16)
    shear_viscosity: 5e-3,
    bulk_viscosity: 1e-2,
    specific_heat: SPECIFIC_HEAT_BONE,
    thermal_conductivity: THERMAL_CONDUCTIVITY_SKULL,
    // α = k/(ρ·cp) = 0.40 / (1920 × 1313) = 1.587e-7 m²/s
    thermal_diffusivity: 1.587e-7,
    perfusion_rate: 5.0, // Much lower perfusion in bone
    arterial_temperature: BODY_TEMPERATURE_C,
    metabolic_heat: 0.1, // Lower metabolic rate
    optical_absorption: 5.0,
    optical_scattering: 200.0,
    refractive_index: 1.65,
    reference_temperature: BODY_TEMPERATURE_C,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

// ============================================================================
// Liver Tissue
// ============================================================================

/// Liver (hepatic tissue) (37°C)
/// Source: Duck (1990), Table 3.3; Table 4.6: c=1578 m/s, ρ=1060 kg/m³
/// Z = ρ·c = 1060 × 1578 = 1 672 680 Pa·s/m
pub const LIVER: TissueProperties = TissueProperties {
    sound_speed: SOUND_SPEED_LIVER,
    density: DENSITY_LIVER,
    impedance: 1_672_680.0,
    absorption_coefficient: 0.4,
    absorption_exponent: 1.0,
    nonlinearity_parameter: B_OVER_A_LIVER, // 6.75 (Duck 1990 Table 4.16 mean)
    shear_viscosity: VISCOSITY_PARENCHYMAL_TISSUE,
    bulk_viscosity: 5e-3,
    specific_heat: SPECIFIC_HEAT_LIVER,
    thermal_conductivity: THERMAL_CONDUCTIVITY_LIVER,
    // α = k/(ρ·cp) = 0.56 / (1060 × 3540) = 1.492e-7 m²/s
    thermal_diffusivity: 1.492e-7,
    perfusion_rate: 100.0, // mL/100g/min
    arterial_temperature: BODY_TEMPERATURE_C,
    metabolic_heat: 0.8, // W/kg
    optical_absorption: 3.0,
    optical_scattering: 120.0,
    refractive_index: REFRACTIVE_INDEX_SOFT_TISSUE_NIR,
    reference_temperature: BODY_TEMPERATURE_C,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

// ============================================================================
// Kidney Tissue
// ============================================================================

/// Kidney cortex (37°C)
/// Source: Duck (1990), Table 3.3
pub const KIDNEY_CORTEX: TissueProperties = TissueProperties {
    sound_speed: SOUND_SPEED_KIDNEY,
    density: DENSITY_TISSUE,
    impedance: 1638000.0,
    absorption_coefficient: 0.5,
    absorption_exponent: 1.0,
    nonlinearity_parameter: B_OVER_A_KIDNEY, // 7.2 (Duck 1990 Table 4.16)
    shear_viscosity: VISCOSITY_PARENCHYMAL_TISSUE,
    bulk_viscosity: 5e-3,
    specific_heat: SPECIFIC_HEAT_TISSUE,
    thermal_conductivity: THERMAL_CONDUCTIVITY_KIDNEY,
    // α = k/(ρ·cp) = 0.50 / (1050 × 3600) = 1.323e-7 m²/s
    thermal_diffusivity: 1.323e-7,
    perfusion_rate: 120.0, // Very high perfusion
    arterial_temperature: BODY_TEMPERATURE_C,
    metabolic_heat: 1.0,
    optical_absorption: 2.5,
    optical_scattering: 110.0,
    refractive_index: REFRACTIVE_INDEX_BRAIN_TISSUE,
    reference_temperature: BODY_TEMPERATURE_C,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

/// Kidney medulla (37°C)
pub const KIDNEY_MEDULLA: TissueProperties = TissueProperties {
    sound_speed: 1565.0,
    density: 1055.0,
    // Z = ρ·c = 1055 × 1565 = 1 651 075 Pa·s/m
    impedance: 1_651_075.0,
    absorption_coefficient: 0.5,
    absorption_exponent: 1.0,
    nonlinearity_parameter: B_OVER_A_KIDNEY, // 7.2 (Duck 1990 Table 4.16)
    shear_viscosity: VISCOSITY_PARENCHYMAL_TISSUE,
    bulk_viscosity: 5e-3,
    specific_heat: SPECIFIC_HEAT_TISSUE,
    thermal_conductivity: THERMAL_CONDUCTIVITY_KIDNEY,
    // α = k/(ρ·cp) = 0.50 / (1055 × 3600) = 1.317e-7 m²/s
    thermal_diffusivity: 1.317e-7,
    perfusion_rate: 130.0,
    arterial_temperature: BODY_TEMPERATURE_C,
    metabolic_heat: 1.1,
    optical_absorption: 2.5,
    optical_scattering: 110.0,
    refractive_index: REFRACTIVE_INDEX_BRAIN_TISSUE,
    reference_temperature: BODY_TEMPERATURE_C,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

// ============================================================================
// Other Common Tissues
// ============================================================================

/// Blood (37°C)
/// Source: Duck (1990), Table 4.6: c=1584 m/s, ρ=1060 kg/m³
/// Z = ρ·c = 1060 × 1584 = 1 679 040 Pa·s/m
pub const BLOOD: TissueProperties = TissueProperties {
    sound_speed: SOUND_SPEED_BLOOD,
    density: DENSITY_BLOOD,
    impedance: 1_679_040.0,
    absorption_coefficient: 0.15,
    absorption_exponent: 1.0,
    nonlinearity_parameter: B_OVER_A_BLOOD, // 6.1 (Duck 1990 Table 4.16)
    shear_viscosity: 4e-3,
    bulk_viscosity: 0.0,
    specific_heat: SPECIFIC_HEAT_BLOOD,
    thermal_conductivity: THERMAL_CONDUCTIVITY_BLOOD,
    // α = k/(ρ·cp) = 0.52 / (1060 × 3617) = 1.356e-7 m²/s
    thermal_diffusivity: 1.356e-7,
    perfusion_rate: 0.0, // Blood IS the perfusion medium
    arterial_temperature: BODY_TEMPERATURE_C,
    metabolic_heat: 0.0,
    optical_absorption: 100.0, // Blood strongly absorbs light
    optical_scattering: 500.0, // High scattering
    refractive_index: REFRACTIVE_INDEX_BIOLOGICAL_FLUID,
    reference_temperature: BODY_TEMPERATURE_C,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

/// Muscle (37°C)
/// Source: Duck (1990) Table 4.1/4.6; IT'IS Foundation v4.0
/// Z = ρ·c = 1090 × 1580 = 1 722 200 Pa·s/m
pub const MUSCLE: TissueProperties = TissueProperties {
    sound_speed: SOUND_SPEED_MUSCLE,
    density: DENSITY_MUSCLE,
    impedance: 1_722_200.0,
    absorption_coefficient: 0.13,
    absorption_exponent: 1.0,
    nonlinearity_parameter: B_OVER_A_MUSCLE, // 7.4 (Duck 1990 Table 4.16)
    shear_viscosity: VISCOSITY_PARENCHYMAL_TISSUE,
    bulk_viscosity: 5e-3,
    specific_heat: SPECIFIC_HEAT_MUSCLE,
    thermal_conductivity: THERMAL_CONDUCTIVITY_MUSCLE,
    // α = k/(ρ·cp) = 0.49 / (1090 × 3421) = 1.314e-7 m²/s
    thermal_diffusivity: 1.314e-7,
    perfusion_rate: 80.0,
    arterial_temperature: BODY_TEMPERATURE_C,
    metabolic_heat: 1.2, // Active metabolism
    optical_absorption: 0.1,
    optical_scattering: 50.0,
    refractive_index: REFRACTIVE_INDEX_SOFT_TISSUE_NIR,
    reference_temperature: BODY_TEMPERATURE_C,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

/// Fat (adipose tissue) (37°C)
/// Source: Duck (1990) Table 4.1/4.6; Z = ρ·c = 928 × 1450 = 1 345 600 Pa·s/m
pub const FAT: TissueProperties = TissueProperties {
    sound_speed: SOUND_SPEED_FAT,
    density: DENSITY_FAT,
    impedance: 1_345_600.0,
    absorption_coefficient: 0.48,
    absorption_exponent: 1.0,
    nonlinearity_parameter: B_OVER_A_FAT, // 9.6 (Duck 1990 Table 4.16)
    shear_viscosity: VISCOSITY_PARENCHYMAL_TISSUE,
    bulk_viscosity: 5e-3,
    specific_heat: SPECIFIC_HEAT_FAT,
    thermal_conductivity: THERMAL_CONDUCTIVITY_FAT,
    // α = k/(ρ·cp) = 0.21 / (928 × 2348) = 9.637e-8 m²/s
    thermal_diffusivity: 9.637e-8,
    perfusion_rate: 30.0, // Lower perfusion
    arterial_temperature: BODY_TEMPERATURE_C,
    metabolic_heat: 0.2,      // Lower metabolic rate
    optical_absorption: 0.01, // Transparent
    optical_scattering: 10.0,
    refractive_index: 1.46,
    reference_temperature: BODY_TEMPERATURE_C,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

/// Cerebrospinal fluid (CSF) (37°C)
pub const CSF: TissueProperties = TissueProperties {
    sound_speed: 1515.0,
    density: 1007.0,
    // Z = ρ·c = 1007 × 1515 = 1 525 605 Pa·s/m
    impedance: 1_525_605.0,
    absorption_coefficient: 0.0,
    absorption_exponent: 1.0,
    nonlinearity_parameter: B_OVER_A_CSF,
    shear_viscosity: 0.7e-3,
    bulk_viscosity: 0.0,
    specific_heat: SPECIFIC_HEAT_CSF,
    thermal_conductivity: THERMAL_CONDUCTIVITY_CSF,
    // α = k/(ρ·cp) = 0.60 / (1007 × 3900) = 1.528e-7 m²/s
    thermal_diffusivity: 1.528e-7,
    perfusion_rate: 0.0, // Not perfused tissue
    arterial_temperature: BODY_TEMPERATURE_C,
    metabolic_heat: 0.0,
    optical_absorption: 0.0,
    optical_scattering: 0.0,
    refractive_index: 1.334,
    reference_temperature: BODY_TEMPERATURE_C,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::numerical::MHZ_TO_HZ;

    #[test]
    fn test_tissue_properties_valid() {
        for (tissue, name) in [
            (&WATER, "WATER"),
            (&BRAIN_WHITE_MATTER, "BRAIN_WHITE_MATTER"),
            (&SKULL, "SKULL"),
            (&LIVER, "LIVER"),
            (&BLOOD, "BLOOD"),
        ] {
            tissue
                .validate()
                .unwrap_or_else(|e| panic!("{name} validation failed: {e:?}"));
        }
    }

    #[test]
    fn test_impedance_calculation() {
        let expected_water = DENSITY_WATER * SOUND_SPEED_WATER;
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
        let att_1mhz = BRAIN_WHITE_MATTER.absorption_at_frequency(MHZ_TO_HZ);
        let att_3mhz = BRAIN_WHITE_MATTER.absorption_at_frequency(3.0 * MHZ_TO_HZ);
        assert!(att_3mhz > att_1mhz);
    }
}
