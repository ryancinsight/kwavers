//! Tissue-property extraction and classification tests.

use super::super::*;
use leto::Array3;
use std::collections::HashMap;

#[test]
fn test_tissue_property_extraction() {
    let fused_result = FusedImageResult {
        intensity_image: Array3::<f64>::from_elem([4, 4, 2], 0.7),
        tissue_properties: HashMap::new(),
        confidence_map: Array3::<f64>::ones([4, 4, 2]),
        uncertainty_map: Some(Array3::<f64>::from_elem([4, 4, 2], 0.1)),
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
    assert_eq!(properties["tissue_classification"].shape(), [4, 4, 2]);
    assert_eq!(properties["oxygenation_index"].shape(), [4, 4, 2]);
    assert_eq!(properties["composite_stiffness"].shape(), [4, 4, 2]);
}

#[test]
fn test_tissue_classification_thresholds() {
    let mut intensity = Array3::<f64>::zeros([4, 4, 2]);
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
    let intensity = Array3::<f64>::from_shape_fn([4, 4, 2], |[i, j, k]| (i + j + k) as f64 / 10.0);

    let oxygenation = properties::compute_oxygenation_index(&intensity);

    // All values should be in [0, 1] range
    for value in oxygenation.iter() {
        assert!(*value >= 0.0);
        assert!(*value <= 1.0);
    }
}

#[test]
fn test_composite_stiffness_range() {
    let intensity = Array3::<f64>::from_shape_fn([4, 4, 2], |[i, j, k]| (i + j + k) as f64 / 10.0);

    let stiffness = properties::compute_composite_stiffness(&intensity);

    // Stiffness should be in expected range [20, 60] kPa
    for value in stiffness.iter() {
        assert!(*value >= 20.0);
        assert!(*value <= 60.0);
    }
}
