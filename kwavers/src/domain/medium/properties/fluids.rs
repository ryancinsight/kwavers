//! Fluid property catalogs
//!
//! Provides standardized fluid property definitions for:
//! - Biological fluids (blood, cerebrospinal fluid)
//! - Coupling fluids for ultrasound transducers
//! - Contrast agents and suspensions
//!
//! Sources:
//! - Duck (1990) - Physical Properties of Tissues
//! - Perry & Green (2007) - Chemical Engineering Handbook
//! - IEC 61161:2013 - Ultrasound equipment safety standard
//! - Gordon et al. (2009) - Acoustic and thermal properties of blood at body temperature
//!
//! Temperature: 37°C (body temperature) unless otherwise noted
//! Pressure: 1 atm unless otherwise noted

use crate::core::constants::fundamental::{
    ATMOSPHERIC_PRESSURE, B_OVER_A_BLOOD, B_OVER_A_WATER, DENSITY_BLOOD, DENSITY_TISSUE,
    SOUND_SPEED_BLOOD,
};
use crate::core::constants::medical::BLOOD_SPECIFIC_HEAT;
use crate::core::constants::thermodynamic::BODY_TEMPERATURE_C;
use super::material::AcousticMaterialProperties;

/// Fluid material properties type alias
pub type FluidProperties = AcousticMaterialProperties;

// ============================================================================
// Biological Fluids
// ============================================================================

/// Blood plasma at 37°C
/// Source: Duck (1990), Gordon et al. (2009)
/// Acoustic properties similar to water with slight frequency dependence
pub const BLOOD_PLASMA: FluidProperties = FluidProperties {
    sound_speed: SOUND_SPEED_BLOOD,
    density: 1026.0,
    impedance: 1624784.0,
    absorption_coefficient: 0.015,
    absorption_exponent: 1.2,
    nonlinearity_parameter: B_OVER_A_WATER, // 5.2 (Duck 1990 Table 4.16)
    shear_viscosity: 1.2e-3,
    bulk_viscosity: 0.0,
    specific_heat: 3840.0,
    thermal_conductivity: 0.55,
    thermal_diffusivity: 1.40e-7,
    perfusion_rate: 100.0, // Blood circulation rate
    arterial_temperature: BODY_TEMPERATURE_C,
    metabolic_heat: 0.0, // Carries heat via circulation
    optical_absorption: 0.1,
    optical_scattering: 10.0,
    refractive_index: 1.335,
    reference_temperature: BODY_TEMPERATURE_C,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

/// Whole blood at 37°C (hematocrit 45%)
/// Source: Duck (1990), Gordon et al. (2009)
/// Contains red blood cells affecting acoustic properties
pub const WHOLE_BLOOD: FluidProperties = FluidProperties {
    sound_speed: 1557.0,
    density: DENSITY_BLOOD,
    impedance: 1650420.0,
    absorption_coefficient: 0.025,
    absorption_exponent: 1.3,
    nonlinearity_parameter: B_OVER_A_BLOOD, // 6.1 (Duck 1990 Table 4.16)
    shear_viscosity: 3.5e-3, // Non-Newtonian: shear-thinning
    bulk_viscosity: 0.0,
    specific_heat: BLOOD_SPECIFIC_HEAT,
    thermal_conductivity: 0.52,
    thermal_diffusivity: 1.35e-7,
    perfusion_rate: 100.0,
    arterial_temperature: BODY_TEMPERATURE_C,
    metabolic_heat: 0.0,
    optical_absorption: 0.5, // Red blood cells scatter and absorb
    optical_scattering: 50.0,
    refractive_index: 1.335,
    reference_temperature: BODY_TEMPERATURE_C,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

/// Cerebrospinal fluid (CSF) at 37°C
/// Source: Duck (1990)
/// Similar to plasma but with slightly different composition
pub const CSF: FluidProperties = FluidProperties {
    sound_speed: 1509.0,
    density: 1007.0,
    impedance: 1520563.0,
    absorption_coefficient: 0.008,
    absorption_exponent: 1.1,
    nonlinearity_parameter: 5.0,
    shear_viscosity: 0.7e-3, // Lower viscosity than blood
    bulk_viscosity: 0.0,
    specific_heat: 3570.0,
    thermal_conductivity: 0.60,
    thermal_diffusivity: 1.67e-7,
    perfusion_rate: 0.0,
    arterial_temperature: BODY_TEMPERATURE_C,
    metabolic_heat: 0.0,
    optical_absorption: 0.05,
    optical_scattering: 5.0,
    refractive_index: 1.333,
    reference_temperature: BODY_TEMPERATURE_C,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

/// Urine at 37°C
/// Source: Duck (1990)
pub const URINE: FluidProperties = FluidProperties {
    sound_speed: 1541.0,
    density: 1005.0,
    // Z = ρ·c = 1005 × 1541 = 1 548 705 Pa·s/m
    impedance: 1_548_705.0,
    absorption_coefficient: 0.012,
    absorption_exponent: 1.15,
    nonlinearity_parameter: 5.1,
    shear_viscosity: 0.95e-3,
    bulk_viscosity: 0.0,
    specific_heat: 3680.0,
    thermal_conductivity: 0.61,
    // α = k/(ρ·cp) = 0.61 / (1005 × 3680) = 1.649e-7 m²/s
    thermal_diffusivity: 1.649e-7,
    perfusion_rate: 0.0,
    arterial_temperature: BODY_TEMPERATURE_C,
    metabolic_heat: 0.0,
    optical_absorption: 0.2,
    optical_scattering: 15.0,
    refractive_index: 1.335,
    reference_temperature: BODY_TEMPERATURE_C,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

// ============================================================================
// Coupling and Contact Fluids
// ============================================================================

/// Ultrasound gel (acoustic coupling medium)
/// Source: Perry & Green (2007)
/// Typical commercial formulation based on mineral oil and thickening agents
pub const ULTRASOUND_GEL: FluidProperties = FluidProperties {
    sound_speed: 1550.0,
    density: 1020.0,
    impedance: 1581000.0,
    absorption_coefficient: 0.008,
    absorption_exponent: 1.1,
    nonlinearity_parameter: 5.0,
    shear_viscosity: 5.0, // Highly viscous for contact
    bulk_viscosity: 0.0,
    specific_heat: 3300.0,
    thermal_conductivity: 0.15,
    // α = k/(ρ·cp) = 0.15 / (1020 × 3300) = 4.456e-8 m²/s
    thermal_diffusivity: 4.456e-8,
    perfusion_rate: 0.0,
    arterial_temperature: 20.0,
    metabolic_heat: 0.0,
    optical_absorption: 0.1,
    optical_scattering: 20.0,
    refractive_index: 1.45,
    reference_temperature: 20.0,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

/// Mineral oil (acoustic coupling medium)
/// Source: Perry & Green (2007)
/// Pure liquid medium without polymer additives
pub const MINERAL_OIL: FluidProperties = FluidProperties {
    sound_speed: 1450.0,
    density: 870.0,
    impedance: 1261500.0,
    absorption_coefficient: 0.005,
    absorption_exponent: 1.0,
    nonlinearity_parameter: 4.5,
    shear_viscosity: 80.0e-3, // Low viscosity
    bulk_viscosity: 0.0,
    specific_heat: 2100.0,
    thermal_conductivity: 0.14,
    thermal_diffusivity: 7.62e-8,
    perfusion_rate: 0.0,
    arterial_temperature: 20.0,
    metabolic_heat: 0.0,
    optical_absorption: 0.01,
    optical_scattering: 5.0,
    refractive_index: 1.47,
    reference_temperature: 20.0,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

/// Distilled water at 37°C
///
/// Sources:
/// - Del Grosso VA, Mader CW (1972). "Speed of sound in pure water."
///   *J. Acoust. Soc. Am.* **52**(5):1442–1446.  c(37 °C) = 1524.0 m/s.
/// - Duck FA (1990). *Physical Properties of Tissue*. Academic Press, Table 2.1.
/// - IEC 61161:2013 (reference conditions for acoustic power measurements).
///
/// Note: 1497 m/s is the value at ~25 °C, not 37 °C.  Water sound speed
/// increases monotonically with temperature up to ≈74 °C; the physiological
/// value is 1524 m/s.
///
/// Impedance Z = ρ·c = 993.3 × 1524 = 1 513 789 Pa·s/m.
pub const WATER_37C: FluidProperties = FluidProperties {
    sound_speed: 1524.0,
    density: 993.3,
    impedance: 1_513_789.0,
    absorption_coefficient: 0.002,
    absorption_exponent: 2.0,
    nonlinearity_parameter: 5.0,
    shear_viscosity: 0.7e-3,
    bulk_viscosity: 0.0,
    specific_heat: 4180.0,
    thermal_conductivity: 0.63,
    thermal_diffusivity: 1.52e-7,
    perfusion_rate: 0.0,
    arterial_temperature: BODY_TEMPERATURE_C,
    metabolic_heat: 0.0,
    optical_absorption: 0.0,
    optical_scattering: 0.0,
    refractive_index: 1.33,
    reference_temperature: BODY_TEMPERATURE_C,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

// ============================================================================
// Contrast Agents and Suspensions
// ============================================================================

/// Microbubble contrast agent (typical formulation)
/// Source: Stride & Saffari (2003) - Microbubble suspensions for contrast imaging
/// Mostly water with suspended gas bubbles (~3-10 μm diameter)
/// Effective properties depend on bubble concentration
pub const MICROBUBBLE_SUSPENSION: FluidProperties = FluidProperties {
    sound_speed: 1480.0,
    density: 1010.0,
    impedance: 1494800.0,
    absorption_coefficient: 0.05, // Higher absorption due to bubble resonance
    absorption_exponent: 1.5,
    nonlinearity_parameter: B_OVER_A_WATER, // 5.2 (Duck 1990 Table 4.16)
    shear_viscosity: 0.8e-3,
    bulk_viscosity: 0.0,
    specific_heat: 4170.0,
    thermal_conductivity: 0.60,
    // α = k/(ρ·cp) = 0.60 / (1010 × 4170) = 1.425e-7 m²/s
    thermal_diffusivity: 1.425e-7,
    perfusion_rate: 0.0,
    arterial_temperature: BODY_TEMPERATURE_C,
    metabolic_heat: 0.0,
    optical_absorption: 0.2,
    optical_scattering: 30.0,
    refractive_index: 1.33,
    reference_temperature: BODY_TEMPERATURE_C,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

/// Nanoparticle suspension (iron oxide or gold nanoparticles)
/// Source: Stride & Saffari (2003)
/// Water-based carrier with suspended nanoparticles for theranostics
pub const NANOPARTICLE_SUSPENSION: FluidProperties = FluidProperties {
    sound_speed: 1490.0,
    density: DENSITY_TISSUE,
    impedance: 1563450.0,
    absorption_coefficient: 0.03,
    absorption_exponent: 1.2,
    nonlinearity_parameter: 5.3,
    shear_viscosity: 1.0e-3,
    bulk_viscosity: 0.0,
    specific_heat: 4150.0,
    thermal_conductivity: 0.59,
    thermal_diffusivity: 1.36e-7,
    perfusion_rate: 0.0,
    arterial_temperature: BODY_TEMPERATURE_C,
    metabolic_heat: 0.0,
    optical_absorption: 1.0, // Significant optical absorption
    optical_scattering: 100.0,
    refractive_index: 1.34,
    reference_temperature: BODY_TEMPERATURE_C,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;

    #[test]
    fn test_blood_properties() {
        assert!(WHOLE_BLOOD.sound_speed > SOUND_SPEED_WATER_SIM);
        assert!(WHOLE_BLOOD.sound_speed < 1600.0);
        WHOLE_BLOOD.validate().unwrap();
    }

    #[test]
    fn test_fluid_impedance_matching() {
        // Blood and tissue should have similar impedance for coupling
        let tissue_impedance = 1_620_000.0; // Brain tissue typical
        let blood_impedance = WHOLE_BLOOD.impedance;
        let mismatch = (blood_impedance - tissue_impedance).abs() / tissue_impedance;
        assert!(mismatch < 0.1); // Less than 10% mismatch
    }

    #[test]
    fn test_coupling_fluid_acoustic_properties() {
        ULTRASOUND_GEL.validate().unwrap();
        MINERAL_OIL.validate().unwrap();

        // Gel and oil should have similar sound speeds for coupling
        let speed_diff = (ULTRASOUND_GEL.sound_speed - MINERAL_OIL.sound_speed).abs();
        assert!(speed_diff < 150.0); // Within 150 m/s
    }

    #[test]
    fn test_water_temperature_dependence() {
        // Water sound speed increases monotonically with temperature up to ~74 °C.
        // At 20 °C: ~1483 m/s; at 37 °C: ~1524 m/s (Del Grosso & Mader 1972).
        assert!(WATER_37C.sound_speed > 1515.0);
        assert!(WATER_37C.sound_speed < 1535.0);
    }

    #[test]
    fn test_contrast_agent_acoustic_differences() {
        // Microbubbles should have higher absorption
        assert!(MICROBUBBLE_SUSPENSION.absorption_coefficient > WATER_37C.absorption_coefficient);

        // Nanoparticles should have high optical absorption
        assert!(NANOPARTICLE_SUSPENSION.optical_absorption > WATER_37C.optical_absorption);
    }

    #[test]
    fn test_csf_similarity_to_plasma() {
        // CSF is mostly water, so properties should be close to plasma
        let speed_diff = (CSF.sound_speed - BLOOD_PLASMA.sound_speed).abs();
        let impedance_diff =
            (CSF.impedance - BLOOD_PLASMA.impedance).abs() / BLOOD_PLASMA.impedance;

        assert!(speed_diff < 100.0);
        assert!(impedance_diff < 0.10);
    }

    #[test]
    fn test_reflection_coefficient_blood_tissue() {
        // Reflection between blood and tissue should be minimal
        let r = WHOLE_BLOOD.reflection_coefficient(&super::super::tissue::BRAIN_GRAY_MATTER);
        assert!(r < 0.05); // Less than 5% reflection
    }

    #[test]
    fn test_attenuation_frequency_scaling() {
        // Test frequency-dependent attenuation for all fluids
        let freqs = [1e6, 2e6, 5e6, 10e6]; // 1, 2, 5, 10 MHz

        for fluid in &[WHOLE_BLOOD, CSF, ULTRASOUND_GEL] {
            let mut prev_att = 0.0;
            for &f in &freqs {
                let att = fluid.absorption_at_frequency(f);
                assert!(att >= prev_att); // Monotonically increasing
                prev_att = att;
            }
        }
    }

    #[test]
    fn test_thermal_properties_consistency() {
        // Thermal diffusivity = k / (ρ * c)
        let expected =
            WHOLE_BLOOD.thermal_conductivity / (WHOLE_BLOOD.density * WHOLE_BLOOD.specific_heat);

        // Allow 1% tolerance
        let tolerance = expected * 0.01;
        assert!((WHOLE_BLOOD.thermal_diffusivity - expected).abs() < tolerance);
    }

    #[test]
    fn test_all_fluids_valid() {
        let fluids = vec![
            BLOOD_PLASMA,
            WHOLE_BLOOD,
            CSF,
            URINE,
            ULTRASOUND_GEL,
            MINERAL_OIL,
            WATER_37C,
            MICROBUBBLE_SUSPENSION,
            NANOPARTICLE_SUSPENSION,
        ];

        for fluid in fluids {
            fluid
                .validate()
                .unwrap_or_else(|e| panic!("Fluid validation failed: {e:?}"));
        }
    }
}
