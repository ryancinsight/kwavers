use super::*;
use ndarray::Array3;

// --- Lifecycle Tests ---

#[test]
fn test_multimodal_fusion_creation() {
    let config = FusionConfig::default();
    let fusion = MultiModalFusion::new(config);
    assert_eq!(fusion.num_registered_modalities(), 0);
    assert!(fusion.registered_data.is_empty());
}

#[test]
fn test_register_ultrasound() {
    let config = FusionConfig::default();
    let mut fusion = MultiModalFusion::new(config);

    let data = Array3::<f64>::zeros((8, 8, 4));
    let result = fusion.register_ultrasound(&data);

    assert!(result.is_ok());
    assert_eq!(fusion.num_registered_modalities(), 1);
    assert!(fusion.is_modality_registered("ultrasound"));
}

// --- Negative Tests & Bounds Constraints ---

#[test]
fn test_fuse_insufficient_modalities() {
    let config = FusionConfig::default();
    let fusion = MultiModalFusion::new(config);
    let result = fusion.fuse();
    
    assert!(result.is_err());
    if let Err(KwaversError::Validation(e)) = result {
        assert!(e.to_string().contains("At least two modalities"));
    } else {
        panic!("Expected Validation ConstraintViolation error.");
    }
}

#[test]
fn test_register_optical_validation() {
    let config = FusionConfig::default();
    let mut fusion = MultiModalFusion::new(config);

    // Negative invariant: Optical intensity cannot be negative
    let mut invalid_data = Array3::<f64>::zeros((4, 4, 2));
    invalid_data[[0, 0, 0]] = -1.0;

    let result = fusion.register_optical(&invalid_data, 550e-9);
    assert!(result.is_err(), "Must enforce physical constraint: Intensity >= 0");

    let valid_data = Array3::<f64>::ones((4, 4, 2));
    let result = fusion.register_optical(&valid_data, 550e-9);
    assert!(result.is_ok());
}

#[test]
fn test_robust_bounds_degenerate() {
    let empty_data = Array3::<f64>::zeros((0, 0, 0));
    let (min, max) = utils::compute_robust_bounds(empty_data.view());
    assert_eq!(min, 0.0);
    assert_eq!(max, 0.0);

    let uniform_data = Array3::<f64>::ones((2, 2, 2));
    let (min, max) = utils::compute_robust_bounds(uniform_data.view());
    assert_eq!(min, 1.0);
    assert_eq!(max, 2.0); 
}

// --- Algorithmic & Property Tests ---

#[test]
fn test_weighted_average_fusion_is_bounded() {
    let config = FusionConfig::default();
    let mut fusion = MultiModalFusion::new(config);

    let shape = (4, 4, 4);
    
    // Bounds check property test: Output must strictly be within [min(A_i), max(B_i)]
    let v_us = 1.0;
    let v_pa = 5.0;

    fusion.register_ultrasound(&Array3::from_elem(shape, v_us)).unwrap();
    fusion.registered_data.insert(
        "photoacoustic".to_string(),
        RegisteredModality {
            data: Array3::from_elem(shape, v_pa),
            quality_score: 0.8,
        },
    );

    let fused = fusion.fuse().unwrap();
    assert_eq!(fused.intensity_image.dim(), shape);

    let weights = &fusion.config.modality_weights;
    let w_us = weights.get("ultrasound").copied().unwrap_or(1.0);
    let w_pa = weights.get("photoacoustic").copied().unwrap_or(1.0);
    let expected = (w_us * v_us + w_pa * v_pa) / (w_us + w_pa);

    for &v in fused.intensity_image.iter() {
        assert!((v - expected).abs() < 1e-9);
        // Property: result bounded
        assert!(v >= v_us && v <= v_pa);
    }
}

#[test]
fn test_maximum_likelihood_fusion_monotonicity() {
    let config = FusionConfig {
        fusion_method: FusionMethod::MaximumLikelihood,
        ..Default::default()
    };
    let mut fusion = MultiModalFusion::new(config);

    let shape = (5, 5, 2);

    // M-step should adaptively trust the cleaner signal
    fusion.registered_data.insert(
        "modality_clean".to_string(),
        RegisteredModality {
            data: Array3::from_elem(shape, 2.0),
            quality_score: 0.99, // Should have low variance initially
        },
    );

    fusion.registered_data.insert(
        "modality_noisy".to_string(),
        RegisteredModality {
            data: Array3::from_elem(shape, 10.0),
            quality_score: 0.01, // High variance
        },
    );

    let fused = fusion.fuse().unwrap();
    
    // Fused result should heavily favor the 'clean' signal
    for &v in fused.intensity_image.iter() {
        assert!(v < 3.0, "MLE should weight higher quality measurement higher");
    }
    
    // Uncertainty map must be generated
    assert!(fused.uncertainty_map.is_some());
    for &u in fused.uncertainty_map.as_ref().unwrap().iter() {
        assert!(u > 0.0, "Uncertainty must be strictly positive");
    }
}

#[test]
fn test_fuse_feature_based_tissue_classification_invariants() {
    let mut config = FusionConfig::default();
    config.fusion_method = FusionMethod::FeatureBased;
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
            &crate::domain::imaging::ultrasound::elastography::ElasticityMap {
                youngs_modulus: Array3::zeros(shape),
                shear_modulus: elasto_data,
                shear_wave_speed: Array3::zeros(shape),
            },
        )
        .unwrap();

    let result = fusion.fuse().unwrap();

    // Tissue classifier must identify combinations with specific properties.
    // Combinations of High Absorption + Low Stiffness => indicative of Blood Vessels (Type 2)
    let classification = result.tissue_properties.get("tissue_classification").unwrap();
    assert_eq!(classification[[0, 0, 0]], 2.0, "Feature-classifier failed specific logic");
}

#[test]
fn test_extract_tissue_properties_generates_composite() {
    let config = FusionConfig::default();
    let mut fusion = MultiModalFusion::new(config);
    let shape = (2, 2, 1);

    fusion.register_ultrasound(&Array3::from_elem(shape, 1.0)).unwrap();
    fusion.registered_data.insert(
        "photoacoustic".to_string(),
        RegisteredModality { data: Array3::from_elem(shape, 2.0), quality_score: 0.8 },
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
