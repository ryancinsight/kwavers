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

use super::material::MaterialProperties;

/// Implant material properties type alias
pub type ImplantProperties = MaterialProperties;

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
    reference_pressure: 101325.0,
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
    reference_pressure: 101325.0,
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
    reference_pressure: 101325.0,
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
    reference_pressure: 101325.0,
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
    reference_pressure: 101325.0,
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
    reference_pressure: 101325.0,
};

/// Polyurethane - Flexible elastomer for coatings and components
/// Source: ASTM F1634
/// Used in artificial heart valves and flexible connectors
pub const POLYURETHANE: ImplantProperties = ImplantProperties {
    sound_speed: 1890.0,
    density: 1050.0,
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
    reference_pressure: 101325.0,
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
    reference_pressure: 101325.0,
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
    reference_pressure: 101325.0,
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
    reference_pressure: 101325.0,
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
    reference_pressure: 101325.0,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metallic_implants_high_impedance() {
        // Metals should have very high acoustic impedance
        assert!(TITANIUM_GRADE5.impedance > 20_000_000.0);
        assert!(STAINLESS_STEEL_316L.impedance > 40_000_000.0);
        assert!(PLATINUM.impedance > 80_000_000.0);
    }

    #[test]
    fn test_polymer_impedance_closer_to_tissue() {
        // Polymers should have impedance closer to tissue than metals
        let tissue_impedance = 1_600_000.0;
        let pmma_ratio = (PMMA.impedance - tissue_impedance).abs() / tissue_impedance;
        let titanium_ratio =
            (TITANIUM_GRADE5.impedance - tissue_impedance).abs() / tissue_impedance;

        assert!(pmma_ratio < titanium_ratio);
    }

    #[test]
    fn test_metallic_implant_thermal_conductivity() {
        // Metals should have high thermal conductivity
        assert!(TITANIUM_GRADE5.thermal_conductivity > 5.0);
        assert!(STAINLESS_STEEL_316L.thermal_conductivity > 10.0);
        assert!(PLATINUM.thermal_conductivity > 50.0);
    }

    #[test]
    fn test_ceramic_acoustic_properties() {
        // Ceramics should have high sound speeds
        assert!(ALUMINA.sound_speed > 10_000.0);
        assert!(ZIRCONIA.sound_speed > 5_000.0);
    }

    #[test]
    fn test_silicone_lower_impedance() {
        // Silicone has lower impedance than tissue
        assert!(SILICONE_RUBBER.impedance < 1_200_000.0);
    }

    #[test]
    fn test_all_implants_valid() {
        let implants = vec![
            TITANIUM_GRADE5,
            STAINLESS_STEEL_316L,
            PLATINUM,
            PMMA,
            UHMWPE,
            SILICONE_RUBBER,
            POLYURETHANE,
            ALUMINA,
            ZIRCONIA,
            CFRP,
            HYDROXYAPATITE,
        ];

        for implant in implants {
            assert!(implant.validate().is_ok(), "Implant validation failed");
        }
    }

    #[test]
    fn test_implant_tissue_reflection() {
        // Test acoustic reflection at implant-tissue interfaces
        let tissue = super::super::tissue::BRAIN_GRAY_MATTER;

        // High impedance mismatch at metal implants
        let r_titanium = tissue.reflection_coefficient(&TITANIUM_GRADE5);
        assert!(r_titanium > 0.9); // >90% reflection

        // Lower mismatch at polymeric implants
        let r_pmma = tissue.reflection_coefficient(&PMMA);
        assert!(r_pmma < 0.4); // <40% reflection
    }

    #[test]
    fn test_composite_properties_between_constituents() {
        // CFRP should have properties between pure carbon (~10k m/s) and resin (~2k m/s)
        assert!(CFRP.sound_speed > 2000.0);
        assert!(CFRP.sound_speed < 5000.0);
    }

    #[test]
    fn test_hydroxyapatite_bone_match() {
        // HA impedance should be closer to bone than pure polymer
        // Typical cortical bone impedance: ~7-8 MΛ
        assert!(HYDROXYAPATITE.impedance > 5_000_000.0);
        assert!(HYDROXYAPATITE.impedance < 15_000_000.0);
    }

    #[test]
    fn test_thermal_diffusivity_consistency() {
        // α = k / (ρ * c) for all materials
        let mats = vec![
            (TITANIUM_GRADE5, "Titanium"),
            (PMMA, "PMMA"),
            (SILICONE_RUBBER, "Silicone"),
            (ALUMINA, "Alumina"),
        ];

        for (mat, name) in mats {
            let expected = mat.thermal_conductivity / (mat.density * mat.specific_heat);
            let tolerance = expected * 0.01;
            assert!(
                (mat.thermal_diffusivity - expected).abs() < tolerance,
                "{}: thermal diffusivity mismatch",
                name
            );
        }
    }

    #[test]
    fn test_optical_opacity_for_metals() {
        // Metals should be optically opaque
        let metals = vec![TITANIUM_GRADE5, STAINLESS_STEEL_316L, PLATINUM];
        for metal in metals {
            assert!(metal.optical_absorption > 50.0);
        }
    }

    #[test]
    fn test_pmma_optical_clarity() {
        // PMMA should have low optical absorption (used in lenses)
        assert!(PMMA.optical_absorption < 1.0);
        assert!(PMMA.refractive_index > 1.4);
    }
}
