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
    let titanium_ratio = (TITANIUM_GRADE5.impedance - tissue_impedance).abs() / tissue_impedance;

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
        implant.validate().unwrap_or_else(|e| panic!("Implant validation failed: {e:?}"));
    }
}

#[test]
fn test_implant_tissue_reflection() {
    // Test acoustic reflection at implant-tissue interfaces
    let tissue = super::super::tissue::BRAIN_GRAY_MATTER;

    // High impedance mismatch at metal implants
    let r_titanium = tissue.reflection_coefficient(&TITANIUM_GRADE5);
    assert!(r_titanium > 0.85); // ~88.7% reflection (high impedance mismatch)

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
        assert!(metal.optical_absorption >= 50.0); // All metals are opaque
    }
}

#[test]
fn test_pmma_optical_clarity() {
    // PMMA should have low optical absorption (used in lenses)
    assert!(PMMA.optical_absorption < 1.0);
    assert!(PMMA.refractive_index > 1.4);
}
