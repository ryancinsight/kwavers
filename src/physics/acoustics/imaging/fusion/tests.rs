//! Integration tests for multi-modal imaging fusion.
//!
//! This module contains integration tests that verify the complete fusion
//! workflow across multiple modules and subsystems.

use super::*;
use ndarray::Array3;
use std::collections::HashMap;

#[test]
fn test_fusion_config_creation_and_defaults() {
    let config = FusionConfig::default();

    assert_eq!(config.output_resolution, [1e-4, 1e-4, 1e-4]);
    assert_eq!(config.fusion_method, FusionMethod::WeightedAverage);
    assert_eq!(config.registration_method, RegistrationMethod::RigidBody);
    assert!(config.uncertainty_quantification);
    assert_eq!(config.confidence_threshold, 0.7);

    // Verify default weights sum to 1.0
    let total_weight: f64 = config.modality_weights.values().sum();
    assert!((total_weight - 1.0).abs() < 1e-10);
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

    // Verify weighted average calculation
    let w_us = fusion.config.modality_weights["ultrasound"];
    let w_pa = fusion.config.modality_weights["photoacoustic"];
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

    // Verify all modalities contributed
    let w_us = fusion.config.modality_weights["ultrasound"];
    let w_pa = fusion.config.modality_weights["photoacoustic"];
    let w_el = fusion.config.modality_weights["elastography"];
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

#[test]
fn test_confidence_map_generation() {
    let config = FusionConfig::default();
    let mut fusion = MultiModalFusion::new(config);

    let shape = (4, 4, 2);

    fusion
        .register_ultrasound(&Array3::from_elem(shape, 1.0))
        .unwrap();
    fusion.registered_data.insert(
        "photoacoustic".to_string(),
        types::RegisteredModality {
            data: Array3::from_elem(shape, 2.0),
            quality_score: 0.9,
        },
    );

    let result = fusion.fuse().unwrap();

    // Confidence map should reflect accumulated weights
    let w_us = fusion.config.modality_weights["ultrasound"];
    let w_pa = fusion.config.modality_weights["photoacoustic"];
    let expected_confidence = w_us + w_pa;

    for value in result.confidence_map.iter() {
        assert!((value - expected_confidence).abs() < 1e-9);
    }
}

#[test]
fn test_uncertainty_quantification_enabled() {
    let config = FusionConfig {
        uncertainty_quantification: true,
        ..Default::default()
    };

    let mut fusion = MultiModalFusion::new(config);
    let shape = (4, 4, 2);

    fusion
        .register_ultrasound(&Array3::from_elem(shape, 1.0))
        .unwrap();
    fusion.registered_data.insert(
        "photoacoustic".to_string(),
        types::RegisteredModality {
            data: Array3::from_elem(shape, 2.0),
            quality_score: 0.8,
        },
    );

    let result = fusion.fuse().unwrap();

    assert!(result.uncertainty_map.is_some());
    let uncertainty = result.uncertainty_map.unwrap();
    assert_eq!(uncertainty.dim(), shape);

    // All values should be non-negative
    for value in uncertainty.iter() {
        assert!(*value >= 0.0);
    }
}

#[test]
fn test_uncertainty_quantification_disabled() {
    let config = FusionConfig {
        uncertainty_quantification: false,
        ..Default::default()
    };

    let mut fusion = MultiModalFusion::new(config);
    let shape = (4, 4, 2);

    fusion
        .register_ultrasound(&Array3::from_elem(shape, 1.0))
        .unwrap();
    fusion.registered_data.insert(
        "photoacoustic".to_string(),
        types::RegisteredModality {
            data: Array3::from_elem(shape, 2.0),
            quality_score: 0.8,
        },
    );

    let result = fusion.fuse().unwrap();

    assert!(result.uncertainty_map.is_none());
}

#[test]
fn test_tissue_property_extraction() {
    let fused_result = FusedImageResult {
        intensity_image: Array3::<f64>::from_elem((4, 4, 2), 0.7),
        tissue_properties: HashMap::new(),
        confidence_map: Array3::<f64>::ones((4, 4, 2)),
        uncertainty_map: Some(Array3::<f64>::from_elem((4, 4, 2), 0.1)),
        registration_transforms: HashMap::new(),
        modality_quality: HashMap::new(),
        coordinates: [vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0]],
    };

    let properties = extract_tissue_properties(&fused_result);

    // Verify all expected properties are present
    assert!(properties.contains_key("tissue_classification"));
    assert!(properties.contains_key("oxygenation_index"));
    assert!(properties.contains_key("composite_stiffness"));

    // Verify property dimensions match input
    assert_eq!(properties["tissue_classification"].dim(), (4, 4, 2));
    assert_eq!(properties["oxygenation_index"].dim(), (4, 4, 2));
    assert_eq!(properties["composite_stiffness"].dim(), (4, 4, 2));
}

#[test]
fn test_tissue_classification_thresholds() {
    let mut intensity = Array3::<f64>::zeros((4, 4, 2));
    intensity[[0, 0, 0]] = 0.2; // Normal
    intensity[[1, 1, 0]] = 0.5; // Borderline
    intensity[[2, 2, 0]] = 0.7; // Moderate abnormality
    intensity[[3, 3, 0]] = 0.9; // High abnormality

    let classification = properties::classify_tissue_types(&intensity);

    assert_eq!(classification[[0, 0, 0]], 0.0);
    assert_eq!(classification[[1, 1, 0]], 0.5);
    assert_eq!(classification[[2, 2, 0]], 1.0);
    assert_eq!(classification[[3, 3, 0]], 2.0);
}

#[test]
fn test_oxygenation_index_range() {
    let intensity = Array3::<f64>::from_shape_fn((4, 4, 2), |(i, j, k)| (i + j + k) as f64 / 10.0);

    let oxygenation = properties::compute_oxygenation_index(&intensity);

    // All values should be in [0, 1] range
    for value in oxygenation.iter() {
        assert!(*value >= 0.0);
        assert!(*value <= 1.0);
    }
}

#[test]
fn test_composite_stiffness_range() {
    let intensity = Array3::<f64>::from_shape_fn((4, 4, 2), |(i, j, k)| (i + j + k) as f64 / 10.0);

    let stiffness = properties::compute_composite_stiffness(&intensity);

    // Stiffness should be in expected range [20, 60] kPa
    for value in stiffness.iter() {
        assert!(*value >= 20.0);
        assert!(*value <= 60.0);
    }
}

#[test]
fn test_coordinate_array_generation() {
    let dims = (10, 8, 5);
    let resolution = [1e-4, 2e-4, 3e-4];

    let coords = registration::generate_coordinate_arrays(dims, resolution);

    assert_eq!(coords[0].len(), dims.0);
    assert_eq!(coords[1].len(), dims.1);
    assert_eq!(coords[2].len(), dims.2);

    // Verify spacing
    assert_eq!(coords[0][0], 0.0);
    assert!((coords[0][1] - resolution[0]).abs() < f64::EPSILON);
    assert!((coords[1][1] - resolution[1]).abs() < f64::EPSILON);
    assert!((coords[2][1] - resolution[2]).abs() < f64::EPSILON);
}

#[test]
fn test_registration_compatibility_validation() {
    // Compatible dimensions
    let result = registration::validate_registration_compatibility((10, 10, 10), (20, 20, 20));
    assert!(result.is_ok());

    // Incompatible dimensions (ratio > 10)
    let result = registration::validate_registration_compatibility((10, 10, 10), (200, 200, 200));
    assert!(result.is_err());
}

#[test]
fn test_bayesian_fusion_single_voxel() {
    let values = vec![1.0, 2.0, 3.0];
    let weights = vec![1.0, 1.0, 1.0];

    let (mean, uncertainty) = quality::bayesian_fusion_single_voxel(&values, &weights);

    // Mean should be 2.0 (average of 1, 2, 3)
    assert!((mean - 2.0).abs() < 1e-10);

    // Uncertainty should be non-negative and <= 1
    assert!(uncertainty >= 0.0);
    assert!(uncertainty <= 1.0);
}

#[test]
fn test_optical_quality_visible_vs_infrared() {
    let intensity = Array3::<f64>::from_elem((4, 4, 2), 100.0);

    let quality_visible = quality::compute_optical_quality(&intensity, 550e-9);
    let quality_infrared = quality::compute_optical_quality(&intensity, 1000e-9);

    // Visible light should have higher quality
    assert!(quality_visible > quality_infrared);
}

#[test]
fn test_affine_transform_composition() {
    let mut transform = AffineTransform::identity();
    transform.translation = [1.0, 2.0, 3.0];
    transform.scale = [2.0, 2.0, 2.0];

    let point = [1.0, 1.0, 1.0];
    let transformed = transform.transform_point(point);

    // Expected: scale first (2, 2, 2), then translate (3, 4, 5)
    assert!((transformed[0] - 3.0).abs() < 1e-10);
    assert!((transformed[1] - 4.0).abs() < 1e-10);
    assert!((transformed[2] - 5.0).abs() < 1e-10);
}

#[test]
fn test_fusion_with_custom_weights() {
    let config = FusionConfig {
        modality_weights: [
            ("ultrasound".to_string(), 0.7),
            ("photoacoustic".to_string(), 0.3),
        ]
        .into_iter()
        .collect(),
        ..Default::default()
    };

    let mut fusion = MultiModalFusion::new(config);
    let shape = (4, 4, 2);

    fusion
        .register_ultrasound(&Array3::from_elem(shape, 1.0))
        .unwrap();
    fusion.registered_data.insert(
        "photoacoustic".to_string(),
        types::RegisteredModality {
            data: Array3::from_elem(shape, 10.0),
            quality_score: 0.8,
        },
    );

    let result = fusion.fuse().unwrap();

    // Expected: (0.7 * 1.0 + 0.3 * 10.0) / 1.0 = 3.7
    let expected = 3.7;
    for value in result.intensity_image.iter() {
        assert!((value - expected).abs() < 1e-9);
    }
}

#[test]
fn test_probabilistic_fusion_uncertainty() {
    let config = FusionConfig {
        fusion_method: FusionMethod::Probabilistic,
        ..Default::default()
    };

    let mut fusion = MultiModalFusion::new(config);
    let shape = (4, 4, 2);

    fusion
        .register_ultrasound(&Array3::from_elem(shape, 1.0))
        .unwrap();
    fusion.registered_data.insert(
        "photoacoustic".to_string(),
        types::RegisteredModality {
            data: Array3::from_elem(shape, 3.0),
            quality_score: 0.8,
        },
    );

    let result = fusion.fuse().unwrap();

    // Probabilistic fusion should always provide uncertainty
    assert!(result.uncertainty_map.is_some());

    let uncertainty = result.uncertainty_map.unwrap();
    assert_eq!(uncertainty.dim(), shape);

    // Should have non-zero uncertainty due to variance between modalities
    for value in uncertainty.iter() {
        assert!(*value > 0.0);
        assert!(*value <= 1.0);
    }
}

#[test]
fn test_registration_transforms_stored() {
    let config = FusionConfig::default();
    let mut fusion = MultiModalFusion::new(config);
    let shape = (4, 4, 2);

    fusion
        .register_ultrasound(&Array3::from_elem(shape, 1.0))
        .unwrap();
    fusion.registered_data.insert(
        "photoacoustic".to_string(),
        types::RegisteredModality {
            data: Array3::from_elem(shape, 2.0),
            quality_score: 0.8,
        },
    );

    let result = fusion.fuse().unwrap();

    // Both modalities should have registration transforms
    assert_eq!(result.registration_transforms.len(), 2);
    assert!(result.registration_transforms.contains_key("ultrasound"));
    assert!(result.registration_transforms.contains_key("photoacoustic"));
}
