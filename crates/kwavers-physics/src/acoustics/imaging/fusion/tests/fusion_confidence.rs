//! Confidence-map and uncertainty-quantification tests.

use super::super::*;
use leto::Array3;

#[test]
fn test_confidence_map_generation() {
    let config = FusionConfig::default();
    let mut fusion = MultiModalFusion::new(config);

    let shape = [4, 4, 2];

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

    // Confidence map should reflect accumulated weights (defaults to 1.0 for each)
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
    let shape = [4, 4, 2];

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

    let uncertainty = result.uncertainty_map.unwrap();
    assert_eq!(uncertainty.shape(), shape);

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
    let shape = [4, 4, 2];

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
