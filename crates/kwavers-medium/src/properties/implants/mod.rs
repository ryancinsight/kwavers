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

use super::material::AcousticMaterialProperties;
use kwavers_core::constants::fundamental::{ATMOSPHERIC_PRESSURE, DENSITY_TISSUE};
use kwavers_core::constants::implants::{
    ACOUSTIC_ABSORPTION_ALUMINA, ACOUSTIC_ABSORPTION_CFRP, ACOUSTIC_ABSORPTION_HYDROXYAPATITE,
    ACOUSTIC_ABSORPTION_PLATINUM, ACOUSTIC_ABSORPTION_PMMA, ACOUSTIC_ABSORPTION_POLYURETHANE,
    ACOUSTIC_ABSORPTION_SILICONE_RUBBER, ACOUSTIC_ABSORPTION_STAINLESS_STEEL_316L,
    ACOUSTIC_ABSORPTION_TITANIUM, ACOUSTIC_ABSORPTION_UHMWPE, ACOUSTIC_ABSORPTION_ZIRCONIA,
    DENSITY_ALUMINA, DENSITY_CFRP, DENSITY_HYDROXYAPATITE, DENSITY_PLATINUM, DENSITY_PMMA,
    DENSITY_SILICONE_RUBBER, DENSITY_STAINLESS_STEEL_316L, DENSITY_TITANIUM_GRADE5, DENSITY_UHMWPE,
    DENSITY_ZIRCONIA, NONLINEARITY_ALUMINA, NONLINEARITY_CFRP, NONLINEARITY_HYDROXYAPATITE,
    NONLINEARITY_PLATINUM, NONLINEARITY_PMMA, NONLINEARITY_POLYURETHANE,
    NONLINEARITY_SILICONE_RUBBER, NONLINEARITY_STAINLESS_STEEL_316L, NONLINEARITY_TITANIUM,
    NONLINEARITY_UHMWPE, NONLINEARITY_ZIRCONIA, SOUND_SPEED_ALUMINA, SOUND_SPEED_CFRP,
    SOUND_SPEED_HYDROXYAPATITE, SOUND_SPEED_PLATINUM, SOUND_SPEED_PMMA, SOUND_SPEED_POLYURETHANE,
    SOUND_SPEED_SILICONE_RUBBER, SOUND_SPEED_STAINLESS_STEEL_316L, SOUND_SPEED_TITANIUM_GRADE5,
    SOUND_SPEED_UHMWPE, SOUND_SPEED_ZIRCONIA, SPECIFIC_HEAT_ALUMINA, SPECIFIC_HEAT_CFRP,
    SPECIFIC_HEAT_HYDROXYAPATITE, SPECIFIC_HEAT_PLATINUM, SPECIFIC_HEAT_PMMA,
    SPECIFIC_HEAT_POLYURETHANE, SPECIFIC_HEAT_SILICONE_RUBBER, SPECIFIC_HEAT_STAINLESS_STEEL_316L,
    SPECIFIC_HEAT_TITANIUM, SPECIFIC_HEAT_UHMWPE, SPECIFIC_HEAT_ZIRCONIA,
    THERMAL_CONDUCTIVITY_ALUMINA, THERMAL_CONDUCTIVITY_CFRP, THERMAL_CONDUCTIVITY_HYDROXYAPATITE,
    THERMAL_CONDUCTIVITY_PLATINUM, THERMAL_CONDUCTIVITY_PMMA, THERMAL_CONDUCTIVITY_POLYURETHANE,
    THERMAL_CONDUCTIVITY_SILICONE_RUBBER, THERMAL_CONDUCTIVITY_STAINLESS_STEEL_316L,
    THERMAL_CONDUCTIVITY_TITANIUM, THERMAL_CONDUCTIVITY_UHMWPE, THERMAL_CONDUCTIVITY_ZIRCONIA,
    THERMAL_DIFFUSIVITY_ALUMINA, THERMAL_DIFFUSIVITY_CFRP, THERMAL_DIFFUSIVITY_HYDROXYAPATITE,
    THERMAL_DIFFUSIVITY_PLATINUM, THERMAL_DIFFUSIVITY_PMMA, THERMAL_DIFFUSIVITY_SILICONE_RUBBER,
    THERMAL_DIFFUSIVITY_STAINLESS_STEEL_316L, THERMAL_DIFFUSIVITY_TITANIUM,
    THERMAL_DIFFUSIVITY_UHMWPE, THERMAL_DIFFUSIVITY_ZIRCONIA,
};
use kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_C;

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
    sound_speed: SOUND_SPEED_TITANIUM_GRADE5,
    density: DENSITY_TITANIUM_GRADE5,
    // Z = ρ·c = 4430 × 6070 = 26 890 100 Pa·s/m
    impedance: 26_890_100.0,
    absorption_coefficient: ACOUSTIC_ABSORPTION_TITANIUM,
    absorption_exponent: 1.0,
    nonlinearity_parameter: NONLINEARITY_TITANIUM,
    shear_viscosity: 0.0,
    bulk_viscosity: 0.0,
    specific_heat: SPECIFIC_HEAT_TITANIUM,
    thermal_conductivity: THERMAL_CONDUCTIVITY_TITANIUM,
    thermal_diffusivity: THERMAL_DIFFUSIVITY_TITANIUM,
    perfusion_rate: 0.0,
    arterial_temperature: BODY_TEMPERATURE_C,
    metabolic_heat: 0.0,
    optical_absorption: 50.0, // Opaque metal
    optical_scattering: 100.0,
    refractive_index: 2.5,
    reference_temperature: BODY_TEMPERATURE_C,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

/// Stainless steel 316L - Standard surgical implant steel
/// Source: ISO 5832-1, ASTM F139
/// Good corrosion resistance, lower cost than titanium
pub const STAINLESS_STEEL_316L: ImplantProperties = ImplantProperties {
    sound_speed: SOUND_SPEED_STAINLESS_STEEL_316L,
    density: DENSITY_STAINLESS_STEEL_316L,
    // Z = ρ·c = 8000 × 5960 = 47 680 000 Pa·s/m
    impedance: 47_680_000.0,
    absorption_coefficient: ACOUSTIC_ABSORPTION_STAINLESS_STEEL_316L,
    absorption_exponent: 1.0,
    nonlinearity_parameter: NONLINEARITY_STAINLESS_STEEL_316L,
    shear_viscosity: 0.0,
    bulk_viscosity: 0.0,
    specific_heat: SPECIFIC_HEAT_STAINLESS_STEEL_316L,
    thermal_conductivity: THERMAL_CONDUCTIVITY_STAINLESS_STEEL_316L,
    thermal_diffusivity: THERMAL_DIFFUSIVITY_STAINLESS_STEEL_316L,
    perfusion_rate: 0.0,
    arterial_temperature: BODY_TEMPERATURE_C,
    metabolic_heat: 0.0,
    optical_absorption: 60.0,
    optical_scattering: 150.0,
    refractive_index: 2.8,
    reference_temperature: BODY_TEMPERATURE_C,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

/// Platinum - High atomic number, excellent biocompatibility
/// Source: ASTM F216
/// Used in pacemakers, catheter tips, and brachytherapy seeds
pub const PLATINUM: ImplantProperties = ImplantProperties {
    sound_speed: SOUND_SPEED_PLATINUM,
    density: DENSITY_PLATINUM,
    // Z = ρ·c = 21450 × 3960 = 84 942 000 Pa·s/m
    impedance: 84_942_000.0,
    absorption_coefficient: ACOUSTIC_ABSORPTION_PLATINUM,
    absorption_exponent: 1.1,
    nonlinearity_parameter: NONLINEARITY_PLATINUM,
    shear_viscosity: 0.0,
    bulk_viscosity: 0.0,
    specific_heat: SPECIFIC_HEAT_PLATINUM,
    thermal_conductivity: THERMAL_CONDUCTIVITY_PLATINUM,
    thermal_diffusivity: THERMAL_DIFFUSIVITY_PLATINUM,
    perfusion_rate: 0.0,
    arterial_temperature: BODY_TEMPERATURE_C,
    metabolic_heat: 0.0,
    optical_absorption: 80.0,
    optical_scattering: 200.0,
    refractive_index: 3.0,
    reference_temperature: BODY_TEMPERATURE_C,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

// ============================================================================
// Polymeric Implants
// ============================================================================

/// PMMA (Polymethyl methacrylate) - Bone cement and lens material
/// Source: ASTM F451
/// Rigid polymer, good optical clarity for some applications
pub const PMMA: ImplantProperties = ImplantProperties {
    sound_speed: SOUND_SPEED_PMMA,
    density: DENSITY_PMMA,
    // Z = ρ·c = 1190 × 2670 = 3 177 300 Pa·s/m
    impedance: 3_177_300.0,
    absorption_coefficient: ACOUSTIC_ABSORPTION_PMMA,
    absorption_exponent: 1.1,
    nonlinearity_parameter: NONLINEARITY_PMMA,
    shear_viscosity: 0.0,
    bulk_viscosity: 0.0,
    specific_heat: SPECIFIC_HEAT_PMMA,
    thermal_conductivity: THERMAL_CONDUCTIVITY_PMMA,
    thermal_diffusivity: THERMAL_DIFFUSIVITY_PMMA,
    perfusion_rate: 0.0,
    arterial_temperature: BODY_TEMPERATURE_C,
    metabolic_heat: 0.0,
    optical_absorption: 0.1,
    optical_scattering: 10.0,
    refractive_index: 1.49,
    reference_temperature: BODY_TEMPERATURE_C,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

/// Ultra-high molecular weight polyethylene (UHMWPE)
/// Source: ASTM F648
/// Used in joint replacement bearing surfaces
pub const UHMWPE: ImplantProperties = ImplantProperties {
    sound_speed: SOUND_SPEED_UHMWPE,
    density: DENSITY_UHMWPE,
    // Z = ρ·c = 935 × 2380 = 2 225 300 Pa·s/m
    impedance: 2_225_300.0,
    absorption_coefficient: ACOUSTIC_ABSORPTION_UHMWPE,
    absorption_exponent: 1.0,
    nonlinearity_parameter: NONLINEARITY_UHMWPE,
    shear_viscosity: 0.0,
    bulk_viscosity: 0.0,
    specific_heat: SPECIFIC_HEAT_UHMWPE,
    thermal_conductivity: THERMAL_CONDUCTIVITY_UHMWPE,
    thermal_diffusivity: THERMAL_DIFFUSIVITY_UHMWPE,
    perfusion_rate: 0.0,
    arterial_temperature: BODY_TEMPERATURE_C,
    metabolic_heat: 0.0,
    optical_absorption: 0.05,
    optical_scattering: 5.0,
    refractive_index: 1.52,
    reference_temperature: BODY_TEMPERATURE_C,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

/// Silicone rubber - Flexible implant material
/// Source: ASTM F381
/// Used in breast implants, seals, and flexible components
pub const SILICONE_RUBBER: ImplantProperties = ImplantProperties {
    sound_speed: SOUND_SPEED_SILICONE_RUBBER,
    density: DENSITY_SILICONE_RUBBER,
    // Z = ρ·c = 970 × 1050 = 1 018 500 Pa·s/m
    impedance: 1_018_500.0,
    absorption_coefficient: ACOUSTIC_ABSORPTION_SILICONE_RUBBER,
    absorption_exponent: 1.2,
    nonlinearity_parameter: NONLINEARITY_SILICONE_RUBBER,
    shear_viscosity: 0.0,
    bulk_viscosity: 0.0,
    specific_heat: SPECIFIC_HEAT_SILICONE_RUBBER,
    thermal_conductivity: THERMAL_CONDUCTIVITY_SILICONE_RUBBER,
    thermal_diffusivity: THERMAL_DIFFUSIVITY_SILICONE_RUBBER,
    perfusion_rate: 0.0,
    arterial_temperature: BODY_TEMPERATURE_C,
    metabolic_heat: 0.0,
    optical_absorption: 1.0,
    optical_scattering: 50.0,
    refractive_index: 1.41,
    reference_temperature: BODY_TEMPERATURE_C,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

/// Polyurethane - Flexible elastomer for coatings and components
/// Source: ASTM F1634
/// Used in artificial heart valves and flexible connectors
pub const POLYURETHANE: ImplantProperties = ImplantProperties {
    sound_speed: SOUND_SPEED_POLYURETHANE,
    density: DENSITY_TISSUE,
    // Z = ρ·c = 1050 × 1890 = 1 984 500 Pa·s/m
    impedance: 1_984_500.0,
    absorption_coefficient: ACOUSTIC_ABSORPTION_POLYURETHANE,
    absorption_exponent: 1.1,
    nonlinearity_parameter: NONLINEARITY_POLYURETHANE,
    shear_viscosity: 0.0,
    bulk_viscosity: 0.0,
    specific_heat: SPECIFIC_HEAT_POLYURETHANE,
    thermal_conductivity: THERMAL_CONDUCTIVITY_POLYURETHANE,
    // α = k/(ρ·c_p) = 0.24 / (1050 × 1800) = 1.270e-7 m²/s
    thermal_diffusivity: 1.270e-7,
    perfusion_rate: 0.0,
    arterial_temperature: BODY_TEMPERATURE_C,
    metabolic_heat: 0.0,
    optical_absorption: 0.3,
    optical_scattering: 20.0,
    refractive_index: 1.48,
    reference_temperature: BODY_TEMPERATURE_C,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

// ============================================================================
// Ceramic Implants
// ============================================================================

/// Alumina (Al₂O₃) - High strength ceramic
/// Source: ASTM F603
/// Used in joint replacement components due to high wear resistance
pub const ALUMINA: ImplantProperties = ImplantProperties {
    sound_speed: SOUND_SPEED_ALUMINA,
    density: DENSITY_ALUMINA,
    // Z = ρ·c = 3970 × 11100 = 44 067 000 Pa·s/m
    impedance: 44_067_000.0,
    absorption_coefficient: ACOUSTIC_ABSORPTION_ALUMINA,
    absorption_exponent: 1.0,
    nonlinearity_parameter: NONLINEARITY_ALUMINA,
    shear_viscosity: 0.0,
    bulk_viscosity: 0.0,
    specific_heat: SPECIFIC_HEAT_ALUMINA,
    thermal_conductivity: THERMAL_CONDUCTIVITY_ALUMINA,
    thermal_diffusivity: THERMAL_DIFFUSIVITY_ALUMINA,
    perfusion_rate: 0.0,
    arterial_temperature: BODY_TEMPERATURE_C,
    metabolic_heat: 0.0,
    optical_absorption: 20.0,
    optical_scattering: 100.0,
    refractive_index: 1.76,
    reference_temperature: BODY_TEMPERATURE_C,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

/// Zirconia (ZrO₂) - High strength ceramic with lower modulus
/// Source: ASTM F1873
/// Superior fracture toughness compared to alumina
pub const ZIRCONIA: ImplantProperties = ImplantProperties {
    sound_speed: SOUND_SPEED_ZIRCONIA,
    density: DENSITY_ZIRCONIA,
    // Z = ρ·c = 6050 × 6000 = 36 300 000 Pa·s/m
    impedance: 36_300_000.0,
    absorption_coefficient: ACOUSTIC_ABSORPTION_ZIRCONIA,
    absorption_exponent: 1.0,
    nonlinearity_parameter: NONLINEARITY_ZIRCONIA,
    shear_viscosity: 0.0,
    bulk_viscosity: 0.0,
    specific_heat: SPECIFIC_HEAT_ZIRCONIA,
    thermal_conductivity: THERMAL_CONDUCTIVITY_ZIRCONIA,
    thermal_diffusivity: THERMAL_DIFFUSIVITY_ZIRCONIA,
    perfusion_rate: 0.0,
    arterial_temperature: BODY_TEMPERATURE_C,
    metabolic_heat: 0.0,
    optical_absorption: 10.0,
    optical_scattering: 80.0,
    refractive_index: 2.15,
    reference_temperature: BODY_TEMPERATURE_C,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

// ============================================================================
// Composite Materials
// ============================================================================

/// Carbon fiber reinforced polymer (CFRP)
/// Source: ASTM E2748
/// High strength-to-weight for structural implants
pub const CFRP: ImplantProperties = ImplantProperties {
    sound_speed: SOUND_SPEED_CFRP,
    density: DENSITY_CFRP,
    // Z = ρ·c = 1600 × 3100 = 4 960 000 Pa·s/m
    impedance: 4_960_000.0,
    absorption_coefficient: ACOUSTIC_ABSORPTION_CFRP,
    absorption_exponent: 1.2,
    nonlinearity_parameter: NONLINEARITY_CFRP,
    shear_viscosity: 0.0,
    bulk_viscosity: 0.0,
    specific_heat: SPECIFIC_HEAT_CFRP,
    thermal_conductivity: THERMAL_CONDUCTIVITY_CFRP,
    thermal_diffusivity: THERMAL_DIFFUSIVITY_CFRP,
    perfusion_rate: 0.0,
    arterial_temperature: BODY_TEMPERATURE_C,
    metabolic_heat: 0.0,
    optical_absorption: 30.0,
    optical_scattering: 80.0,
    refractive_index: 1.6,
    reference_temperature: BODY_TEMPERATURE_C,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};

/// Hydroxyapatite (HA) - Bone-mimetic ceramic
/// Source: ASTM F1185
/// Composition: Ca₁₀(PO₄)₆(OH)₂, closely matched to bone mineral
pub const HYDROXYAPATITE: ImplantProperties = ImplantProperties {
    sound_speed: SOUND_SPEED_HYDROXYAPATITE,
    density: DENSITY_HYDROXYAPATITE,
    // Z = ρ·c = 3220 × 3640 = 11 720 800 Pa·s/m
    impedance: 11_720_800.0,
    absorption_coefficient: ACOUSTIC_ABSORPTION_HYDROXYAPATITE,
    absorption_exponent: 1.1,
    nonlinearity_parameter: NONLINEARITY_HYDROXYAPATITE,
    shear_viscosity: 0.0,
    bulk_viscosity: 0.0,
    specific_heat: SPECIFIC_HEAT_HYDROXYAPATITE,
    thermal_conductivity: THERMAL_CONDUCTIVITY_HYDROXYAPATITE,
    thermal_diffusivity: THERMAL_DIFFUSIVITY_HYDROXYAPATITE,
    perfusion_rate: 0.0,
    arterial_temperature: BODY_TEMPERATURE_C,
    metabolic_heat: 0.0,
    optical_absorption: 50.0,
    optical_scattering: 150.0,
    refractive_index: 1.65,
    reference_temperature: BODY_TEMPERATURE_C,
    reference_pressure: ATMOSPHERIC_PRESSURE,
};
