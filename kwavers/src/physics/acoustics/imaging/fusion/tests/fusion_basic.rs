//! Basic fusion configuration, registration, and weighted-average tests.

use super::super::*;
use ndarray::Array3;

#[test]
fn test_fusion_config_creation_and_defaults() {
    let config = FusionConfig::default();

    assert_eq!(config.output_resolution, [1e-4, 1e-4, 1e-4]);
    assert_eq!(config.fusion_method, FusionMethod::WeightedAverage);
    assert_eq!(config.registration_method, RegistrationMethod::RigidBody);
    assert!(!config.uncertainty_quantification); // Default is false
    assert_eq!(config.min_quality_threshold, 0.3);

    // Default weights are empty (will be set when registering modalities)
    assert_eq!(config.modality_weights.len(), 0);
}

#[test]
fn test_multimodal_fusion_registration() {
    let config = FusionConfig::default();
    let mut fusion = MultiModalFusion::new(config);

    assert_eq!(fusion.num_registered_modalities(), 0);

    // Register ultrasound
    let us_data = Array3::<f64>::from_elem((8, 8, 4), 1.0);
    fusion.register_ultrasound(&us_data).unwrap();

    assert_eq!(fusion.num_registered_modalities(), 1);
    assert!(fusion.is_modality_registered("ultrasound"));
    assert!(!fusion.is_modality_registered("photoacoustic"));
}

#[test]
fn test_weighted_average_fusion_two_modalities() {
    let config = FusionConfig::default();
    let mut fusion = MultiModalFusion::new(config);

    let shape = (8, 8, 4);

    // Register two modalities with known values
    fusion
        .register_ultrasound(&Array3::from_elem(shape, 2.0))
        .unwrap();

    fusion.registered_data.insert(
        "photoacoustic".to_string(),
        types::RegisteredModality {
            data: Array3::from_elem(shape, 4.0),
            quality_score: 0.8,
        },
    );

    let result = fusion.fuse().unwrap();

    // Verify dimensions
    assert_eq!(result.intensity_image.dim(), shape);
    assert_eq!(result.confidence_map.dim(), shape);

    // Verify weighted average calculation (defaults to 1.0 for each modality)
    let w_us = fusion
        .config
        .modality_weights
        .get("ultrasound")
        .copied()
        .unwrap_or(1.0);
    let w_pa = fusion
        .config
        .modality_weights
        .get("photoacoustic")
        .copied()
        .unwrap_or(1.0);
    let expected = (w_us * 2.0 + w_pa * 4.0) / (w_us + w_pa);

    for value in result.intensity_image.iter() {
        assert!((value - expected).abs() < 1e-9);
    }
}

#[test]
fn test_weighted_average_fusion_three_modalities() {
    let config = FusionConfig::default();
    let mut fusion = MultiModalFusion::new(config);

    let shape = (6, 6, 3);

    // Register three modalities
    fusion
        .register_ultrasound(&Array3::from_elem(shape, 1.0))
        .unwrap();

    fusion.registered_data.insert(
        "photoacoustic".to_string(),
        types::RegisteredModality {
            data: Array3::from_elem(shape, 2.0),
            quality_score: 0.85,
        },
    );

    fusion.registered_data.insert(
        "elastography".to_string(),
        types::RegisteredModality {
            data: Array3::from_elem(shape, 3.0),
            quality_score: 0.75,
        },
    );

    let result = fusion.fuse().unwrap();

    assert_eq!(result.intensity_image.dim(), shape);
    assert_eq!(result.modality_quality.len(), 3);

    // Verify all modalities contributed (defaults to 1.0 for each modality)
    let w_us = fusion
        .config
        .modality_weights
        .get("ultrasound")
        .copied()
        .unwrap_or(1.0);
    let w_pa = fusion
        .config
        .modality_weights
        .get("photoacoustic")
        .copied()
        .unwrap_or(1.0);
    let w_el = fusion
        .config
        .modality_weights
        .get("elastography")
        .copied()
        .unwrap_or(1.0);
    let expected = (w_us * 1.0 + w_pa * 2.0 + w_el * 3.0) / (w_us + w_pa + w_el);

    for value in result.intensity_image.iter() {
        assert!((value - expected).abs() < 1e-9);
    }
}

#[test]
fn test_fusion_insufficient_modalities() {
    let config = FusionConfig::default();
    let fusion = MultiModalFusion::new(config.clone());

    // Attempt fusion with no modalities
    let result = fusion.fuse();
    assert!(result.is_err());

    // Attempt fusion with only one modality
    let mut fusion = MultiModalFusion::new(config);
    fusion
        .register_ultrasound(&Array3::zeros((4, 4, 2)))
        .unwrap();
    let result = fusion.fuse();
    assert!(result.is_err());
}
