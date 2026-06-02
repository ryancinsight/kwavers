use super::*;

#[test]
fn test_fuse_feature_based_tissue_classification_invariants() {
    let mut config = FusionConfig::default();
    config.fusion_method = ImagingFusionMethod::FeatureBased;
    config.uncertainty_quantification = false;

    let mut fusion = MultiModalFusion::new(config);
    let shape = (2, 2, 1);

    // US
    let mut us_data = Array3::<f64>::zeros(shape);
    us_data[[0, 0, 0]] = 0.2; // Low scattering
    us_data[[1, 1, 0]] = 1.0; // Anchor for normalization

    // PA
    let mut pa_data = Array3::<f64>::zeros(shape);
    pa_data[[0, 0, 0]] = 0.8; // High absorption
    pa_data[[1, 1, 0]] = 1.0; // Anchor for normalization

    // Elasto
    let mut elasto_data = Array3::<f64>::zeros(shape);
    elasto_data[[0, 0, 0]] = 0.3; // Low stiffness
    elasto_data[[1, 1, 0]] = 1.0; // Anchor for normalization

    fusion.register_ultrasound(&us_data).unwrap();
    fusion.register_photoacoustic(&pa_data).unwrap();
    fusion
        .register_elastography(
            &kwavers_domain::imaging::ultrasound::elastography::ElasticityMap {
                youngs_modulus: Array3::zeros(shape),
                shear_modulus: elasto_data,
                shear_wave_speed: Array3::zeros(shape),
            },
        )
        .unwrap();

    let result = fusion.fuse().unwrap();

    // Tissue classifier must identify combinations with specific properties.
    // Combinations of High Absorption + Low Stiffness => indicative of Blood Vessels (Type 2)
    let classification = result
        .tissue_properties
        .get("tissue_classification")
        .unwrap();
    assert_eq!(
        classification[[0, 0, 0]],
        2.0,
        "Feature-classifier failed specific logic"
    );
}

#[test]
fn test_extract_tissue_properties_generates_composite() {
    let config = FusionConfig::default();
    let mut fusion = MultiModalFusion::new(config);
    let shape = (2, 2, 1);

    fusion
        .register_ultrasound(&Array3::from_elem(shape, 1.0))
        .unwrap();
    fusion.registered_data.insert(
        "photoacoustic".to_string(),
        RegisteredModality {
            data: Array3::from_elem(shape, 2.0),
            quality_score: 0.8,
        },
    );

    let fused = fusion.fuse().unwrap();
    let properties = fusion.extract_tissue_properties(&fused);

    // Validate output dictionary
    assert!(properties.contains_key("tissue_classification"));
    assert!(properties.contains_key("oxygenation_index"));
    assert!(properties.contains_key("composite_stiffness"));

    // Bounds verify composite stiffness extraction logic
    let stiff = properties.get("composite_stiffness").unwrap();
    for &v in stiff.iter() {
        assert!(v >= 0.0);
    }
}
