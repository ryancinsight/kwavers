//! Implant property catalogs
//!
//! Provides standardized material properties for:
//! - Metallic implants (titanium, stainless steel, platinum)
//! - Polymeric implants (PMMA, polyethylene, silicone)
//! - Ceramic implants (zirconia, alumina)
//! - Composite materials used in clinical devices
//!
//! Sources:
//! - Szabo (2004) - Diagnostic Ultrasound Imaging
//! - Duck (1990) - Physical Properties of Tissues
//! - Perry & Green (2007) - Chemical Engineering Handbook
//! - ASTM standards for biomedical materials
//! - ISO 5832 - Metallic materials for surgical implants
//!
//! Temperature: 37°C (body temperature) unless otherwise noted
//! Pressure: 1 atm unless otherwise noted

use crate::core::constants::fundamental::{ATMOSPHERIC_PRESSURE, DENSITY_TISSUE};
use super::material::AcousticMaterialProperties;

#[cfg(test)]
mod tests;

/// Implant material properties type alias
pub type ImplantProperties = AcousticMaterialProperties;

// ============================================================================
// Metallic Implants
// ============================================================================

/// Titanium Grade 5 (Ti-6Al-4V) - Most common surgical implant metal
/// Source: ISO 5832-3, ASTM F136
/// High strength-to-weight ratio, excellent biocompatibility
pub const TITANIUM_GRADE5: ImplantProperties = ImplantProperties {
    sound_speed: 6070.0, // Much higher than tissue
    density: 4430.0,
    impedance: 26_930_100.0,
    absorption_coefficient: 0.1,
    absorption_exponent: 1.0,
    nonlinearity_parameter: 1.5,
    shear_viscosity: 0.0,
    bulk_viscosity: 0.0,
    specific_heat: 560.0,
    thermal_conductivity: 7.4,
    thermal_diffusivity: 2.99e-6,
    perfusion_rate: 0.0,
    arterial_temperature: 37.0,
    metabolic_heat: 0.0,
    optical_absorption: 50.0, // Opaque metal
    optical_scattering: 100.0,
    refractive_index: 2.5,
    reference_temperature: 37.0,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

/// Stainless steel 316L - Standard surgical implant steel
/// Source: ISO 5832-1, ASTM F139
/// Good corrosion resistance, lower cost than titanium
pub const STAINLESS_STEEL_316L: ImplantProperties = ImplantProperties {
    sound_speed: 5960.0,
    density: 8000.0,
    impedance: 47_680_000.0,
    absorption_coefficient: 0.15,
    absorption_exponent: 1.0,
    nonlinearity_parameter: 1.8,
    shear_viscosity: 0.0,
    bulk_viscosity: 0.0,
    specific_heat: 500.0,
    thermal_conductivity: 16.0,
    thermal_diffusivity: 4.0e-6,
    perfusion_rate: 0.0,
    arterial_temperature: 37.0,
    metabolic_heat: 0.0,
    optical_absorption: 60.0,
    optical_scattering: 150.0,
    refractive_index: 2.8,
    reference_temperature: 37.0,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

/// Platinum - High atomic number, excellent biocompatibility
/// Source: ASTM F216
/// Used in pacemakers, catheter tips, and brachytherapy seeds
pub const PLATINUM: ImplantProperties = ImplantProperties {
    sound_speed: 3960.0,
    density: 21_450.0,
    impedance: 85_038_000.0,
    absorption_coefficient: 0.5,
    absorption_exponent: 1.1,
    nonlinearity_parameter: 2.0,
    shear_viscosity: 0.0,
    bulk_viscosity: 0.0,
    specific_heat: 135.0,
    thermal_conductivity: 71.6,
    thermal_diffusivity: 2.46e-5,
    perfusion_rate: 0.0,
    arterial_temperature: 37.0,
    metabolic_heat: 0.0,
    optical_absorption: 80.0,
    optical_scattering: 200.0,
    refractive_index: 3.0,
    reference_temperature: 37.0,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

// ============================================================================
// Polymeric Implants
// ============================================================================

/// PMMA (Polymethyl methacrylate) - Bone cement and lens material
/// Source: ASTM F451
/// Rigid polymer, good optical clarity for some applications
pub const PMMA: ImplantProperties = ImplantProperties {
    sound_speed: 2670.0,
    density: 1190.0,
    impedance: 3_180_300.0,
    absorption_coefficient: 0.08,
    absorption_exponent: 1.1,
    nonlinearity_parameter: 3.0,
    shear_viscosity: 0.0,
    bulk_viscosity: 0.0,
    specific_heat: 1470.0,
    thermal_conductivity: 0.19,
    thermal_diffusivity: 1.08e-7,
    perfusion_rate: 0.0,
    arterial_temperature: 37.0,
    metabolic_heat: 0.0,
    optical_absorption: 0.1,
    optical_scattering: 10.0,
    refractive_index: 1.49,
    reference_temperature: 37.0,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

/// Ultra-high molecular weight polyethylene (UHMWPE)
/// Source: ASTM F648
/// Used in joint replacement bearing surfaces
pub const UHMWPE: ImplantProperties = ImplantProperties {
    sound_speed: 2380.0,
    density: 935.0,
    impedance: 2_224_300.0,
    absorption_coefficient: 0.05,
    absorption_exponent: 1.0,
    nonlinearity_parameter: 2.8,
    shear_viscosity: 0.0,
    bulk_viscosity: 0.0,
    specific_heat: 2300.0,
    thermal_conductivity: 0.42,
    thermal_diffusivity: 1.95e-7,
    perfusion_rate: 0.0,
    arterial_temperature: 37.0,
    metabolic_heat: 0.0,
    optical_absorption: 0.05,
    optical_scattering: 5.0,
    refractive_index: 1.52,
    reference_temperature: 37.0,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

/// Silicone rubber - Flexible implant material
/// Source: ASTM F381
/// Used in breast implants, seals, and flexible components
pub const SILICONE_RUBBER: ImplantProperties = ImplantProperties {
    sound_speed: 1050.0, // Lower speed than tissue
    density: 970.0,
    impedance: 1_018_500.0,
    absorption_coefficient: 0.12,
    absorption_exponent: 1.2,
    nonlinearity_parameter: 4.0,
    shear_viscosity: 0.0,
    bulk_viscosity: 0.0,
    specific_heat: 1500.0,
    thermal_conductivity: 0.25,
    thermal_diffusivity: 1.72e-7,
    perfusion_rate: 0.0,
    arterial_temperature: 37.0,
    metabolic_heat: 0.0,
    optical_absorption: 1.0,
    optical_scattering: 50.0,
    refractive_index: 1.41,
    reference_temperature: 37.0,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

/// Polyurethane - Flexible elastomer for coatings and components
/// Source: ASTM F1634
/// Used in artificial heart valves and flexible connectors
pub const POLYURETHANE: ImplantProperties = ImplantProperties {
    sound_speed: 1890.0,
    density: DENSITY_TISSUE,
    impedance: 1_984_500.0,
    absorption_coefficient: 0.10,
    absorption_exponent: 1.1,
    nonlinearity_parameter: 3.5,
    shear_viscosity: 0.0,
    bulk_viscosity: 0.0,
    specific_heat: 1800.0,
    thermal_conductivity: 0.24,
    thermal_diffusivity: 1.27e-7,
    perfusion_rate: 0.0,
    arterial_temperature: 37.0,
    metabolic_heat: 0.0,
    optical_absorption: 0.3,
    optical_scattering: 20.0,
    refractive_index: 1.48,
    reference_temperature: 37.0,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

// ============================================================================
// Ceramic Implants
// ============================================================================

/// Alumina (Al₂O₃) - High strength ceramic
/// Source: ASTM F603
/// Used in joint replacement components due to high wear resistance
pub const ALUMINA: ImplantProperties = ImplantProperties {
    sound_speed: 11_100.0,
    density: 3970.0,
    impedance: 44_037_000.0,
    absorption_coefficient: 0.05,
    absorption_exponent: 1.0,
    nonlinearity_parameter: 1.2,
    shear_viscosity: 0.0,
    bulk_viscosity: 0.0,
    specific_heat: 880.0,
    thermal_conductivity: 30.0,
    thermal_diffusivity: 8.54e-6,
    perfusion_rate: 0.0,
    arterial_temperature: 37.0,
    metabolic_heat: 0.0,
    optical_absorption: 20.0,
    optical_scattering: 100.0,
    refractive_index: 1.76,
    reference_temperature: 37.0,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

/// Zirconia (ZrO₂) - High strength ceramic with lower modulus
/// Source: ASTM F1873
/// Superior fracture toughness compared to alumina
pub const ZIRCONIA: ImplantProperties = ImplantProperties {
    sound_speed: 6000.0,
    density: 6050.0,
    impedance: 36_300_000.0,
    absorption_coefficient: 0.08,
    absorption_exponent: 1.0,
    nonlinearity_parameter: 1.4,
    shear_viscosity: 0.0,
    bulk_viscosity: 0.0,
    specific_heat: 500.0,
    thermal_conductivity: 2.0,
    thermal_diffusivity: 6.61e-7,
    perfusion_rate: 0.0,
    arterial_temperature: 37.0,
    metabolic_heat: 0.0,
    optical_absorption: 10.0,
    optical_scattering: 80.0,
    refractive_index: 2.15,
    reference_temperature: 37.0,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

// ============================================================================
// Composite Materials
// ============================================================================

/// Carbon fiber reinforced polymer (CFRP)
/// Source: ASTM E2748
/// High strength-to-weight for structural implants
pub const CFRP: ImplantProperties = ImplantProperties {
    sound_speed: 3100.0,
    density: 1600.0,
    impedance: 4_960_000.0,
    absorption_coefficient: 0.15,
    absorption_exponent: 1.2,
    nonlinearity_parameter: 2.5,
    shear_viscosity: 0.0,
    bulk_viscosity: 0.0,
    specific_heat: 900.0,
    thermal_conductivity: 5.0,
    thermal_diffusivity: 3.47e-6,
    perfusion_rate: 0.0,
    arterial_temperature: 37.0,
    metabolic_heat: 0.0,
    optical_absorption: 30.0,
    optical_scattering: 80.0,
    refractive_index: 1.6,
    reference_temperature: 37.0,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

/// Hydroxyapatite (HA) - Bone-mimetic ceramic
/// Source: ASTM F1185
/// Composition: Ca₁₀(PO₄)₆(OH)₂, closely matched to bone mineral
pub const HYDROXYAPATITE: ImplantProperties = ImplantProperties {
    sound_speed: 3640.0,
    density: 3220.0,
    impedance: 11_724_800.0,
    absorption_coefficient: 0.2,
    absorption_exponent: 1.1,
    nonlinearity_parameter: 1.8,
    shear_viscosity: 0.0,
    bulk_viscosity: 0.0,
    specific_heat: 880.0,
    thermal_conductivity: 1.2,
    thermal_diffusivity: 4.21e-7,
    perfusion_rate: 0.0,
    arterial_temperature: 37.0,
    metabolic_heat: 0.0,
    optical_absorption: 50.0,
    optical_scattering: 150.0,
    refractive_index: 1.65,
    reference_temperature: 37.0,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};
