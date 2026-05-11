use super::*;

// ========================================================================
// TissuePropertyMap Composition Tests
// ========================================================================

#[test]
fn test_tissue_property_map_uniform_composition() {
    // Create canonical liver properties
    let liver = AcousticPropertyData::liver();
    let shape = (8, 8, 8);

    // Compose tissue map from canonical type
    let tissue_map = TissuePropertyMap::uniform(shape, liver);

    // Verify shape
    assert_eq!(tissue_map.shape(), shape);
    assert_eq!(tissue_map.ndim(), 3);

    // Verify all elements match source properties
    assert_eq!(tissue_map.speed_of_sound[[0, 0, 0]], liver.sound_speed);
    assert_eq!(tissue_map.density[[4, 4, 4]], liver.density);
    assert_eq!(
        tissue_map.attenuation[[7, 7, 7]],
        liver.absorption_coefficient
    );
    assert_eq!(tissue_map.nonlinearity[[3, 2, 1]], liver.nonlinearity);
}

#[test]
fn test_tissue_property_map_extraction() {
    // Create canonical water properties
    let water = AcousticPropertyData::water();
    let shape = (16, 16, 16);
    let tissue_map = TissuePropertyMap::uniform(shape, water);

    // Extract properties at various locations
    let props_center = tissue_map.at((8, 8, 8)).expect("valid index");
    let props_corner = tissue_map.at((0, 0, 0)).expect("valid index");
    let props_edge = tissue_map.at((15, 15, 15)).expect("valid index");

    // Verify extracted properties match source
    assert_eq!(props_center.density, water.density);
    assert_eq!(props_corner.sound_speed, water.sound_speed);
    assert_eq!(props_edge.nonlinearity, water.nonlinearity);

    // Verify derived quantities are available
    assert!(props_center.impedance() > 0.0);
    // Wavelength = c / f
    let wavelength_1mhz = props_center.sound_speed / 1e6;
    assert!(wavelength_1mhz > 0.0);
}

#[test]
fn test_tissue_property_map_bounds_checking() {
    let tissue_map = TissuePropertyMap::water((10, 10, 10));

    // Valid indices: check they return finite values
    let p000 = tissue_map.at((0, 0, 0)).unwrap();
    assert!(p000.density > 0.0);
    tissue_map.at((9, 9, 9)).unwrap();
    tissue_map.at((5, 5, 5)).unwrap();

    // Out-of-bounds indices should fail
    assert!(tissue_map.at((10, 5, 5)).is_err());
    assert!(tissue_map.at((5, 10, 5)).is_err());
    assert!(tissue_map.at((5, 5, 10)).is_err());
    assert!(tissue_map.at((10, 10, 10)).is_err());
}

#[test]
fn test_tissue_property_map_convenience_constructors() {
    let shape = (12, 12, 12);

    // Test all tissue-specific constructors
    let water_map = TissuePropertyMap::water(shape);
    let liver_map = TissuePropertyMap::liver(shape);
    let brain_map = TissuePropertyMap::brain(shape);
    let kidney_map = TissuePropertyMap::kidney(shape);
    let muscle_map = TissuePropertyMap::muscle(shape);

    // Verify shapes
    assert_eq!(water_map.shape(), shape);
    assert_eq!(liver_map.shape(), shape);
    assert_eq!(brain_map.shape(), shape);
    assert_eq!(kidney_map.shape(), shape);
    assert_eq!(muscle_map.shape(), shape);

    // Verify properties are distinct for different tissues
    let water_props = water_map.at((0, 0, 0)).unwrap();
    let liver_props = liver_map.at((0, 0, 0)).unwrap();
    let brain_props = brain_map.at((0, 0, 0)).unwrap();

    // Different tissues should have different properties
    assert_ne!(water_props.density, liver_props.density);
    assert_ne!(liver_props.sound_speed, brain_props.sound_speed);
}

#[test]
fn test_tissue_property_map_shape_consistency() {
    let shape = (8, 8, 8);
    let tissue_map = TissuePropertyMap::liver(shape);

    // Validation should pass for consistent shapes
    tissue_map.validate_shape_consistency().unwrap();
}

#[test]
fn test_tissue_property_map_round_trip() {
    // Create canonical properties
    let kidney = AcousticPropertyData::kidney();
    let shape = (10, 10, 10);

    // Domain → Physics: construct map
    let tissue_map = TissuePropertyMap::uniform(shape, kidney);

    // Physics → Domain: extract at multiple locations
    for i in 0..10 {
        for j in 0..10 {
            for k in 0..10 {
                let extracted = tissue_map.at((i, j, k)).expect("valid index");

                // Verify round-trip preserves properties
                assert_eq!(extracted.density, kidney.density);
                assert_eq!(extracted.sound_speed, kidney.sound_speed);
                assert_eq!(extracted.nonlinearity, kidney.nonlinearity);

                // Verify derived quantities are consistent
                assert!((extracted.impedance() - kidney.impedance()).abs() < 1e-10);
            }
        }
    }
}

#[test]
fn test_tissue_property_map_heterogeneous_simulation() {
    // Simulate a heterogeneous tissue structure: water background with liver inclusion
    let shape = (16, 16, 16);
    let water = AcousticPropertyData::water();
    let liver = AcousticPropertyData::liver();

    // Start with water background
    let mut tissue_map = TissuePropertyMap::uniform(shape, water);

    // Insert liver inclusion in center region (8x8x8 cube)
    for i in 4..12 {
        for j in 4..12 {
            for k in 4..12 {
                tissue_map.speed_of_sound[[i, j, k]] = liver.sound_speed;
                tissue_map.density[[i, j, k]] = liver.density;
                tissue_map.attenuation[[i, j, k]] = liver.absorption_coefficient;
                tissue_map.nonlinearity[[i, j, k]] = liver.nonlinearity;
            }
        }
    }

    // Verify background (water) properties
    let bg_props = tissue_map.at((0, 0, 0)).expect("valid");
    assert_eq!(bg_props.density, water.density);

    // Verify inclusion (liver) properties
    let inclusion_props = tissue_map.at((8, 8, 8)).expect("valid");
    assert_eq!(inclusion_props.density, liver.density);
    assert_eq!(inclusion_props.sound_speed, liver.sound_speed);

    // Verify shape consistency
    tissue_map.validate_shape_consistency().unwrap();
}

#[test]
fn test_tissue_property_map_clinical_workflow() {
    // Simulate clinical workflow: patient imaging → treatment planning
    let grid_shape = (32, 32, 32);

    // Step 1: Initialize with background tissue (muscle)
    let muscle = AcousticPropertyData::muscle();
    let patient_tissue = TissuePropertyMap::uniform(grid_shape, muscle);

    // Step 2: Extract properties at treatment target
    let target_location = (16, 16, 16);
    let target_props = patient_tissue.at(target_location).expect("valid");

    // Step 3: Calculate treatment parameters using canonical properties
    let acoustic_impedance = target_props.impedance();
    // Wavelength = c / f
    let wavelength_at_1mhz = target_props.sound_speed / 1e6;

    // Verify clinical parameters are physically reasonable
    assert!(acoustic_impedance > 1e6); // Typical tissue impedance > 1 MRayl
    assert!(wavelength_at_1mhz > 1e-3); // Wavelength > 1 mm at 1 MHz
    assert!(wavelength_at_1mhz < 2e-3); // Wavelength < 2 mm for typical tissue
}
