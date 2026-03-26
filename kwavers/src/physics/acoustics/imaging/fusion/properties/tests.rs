use super::classification::{classify_tissue_types, detect_regions_of_interest};
use super::extractor::extract_tissue_properties;
use super::mechanical::{
    compute_composite_stiffness, compute_tissue_density, BASELINE_SOFT_TISSUE_STIFFNESS_KPA,
    DENSITY_NONLINEARITY_EXPONENT, STIFFNESS_INTENSITY_COUPLING,
};
use super::oxygenation::compute_oxygenation_index;
use super::super::types::FusedImageResult;
use ndarray::Array3;
use std::collections::HashMap;

#[test]
fn test_classify_tissue_types_normal() {
    let intensity = Array3::<f64>::from_elem((4, 4, 2), 0.2);
    let classification = classify_tissue_types(&intensity);

    for value in classification.iter() {
        assert_eq!(*value, 0.0); // Normal tissue
    }
}

#[test]
fn test_classify_tissue_types_borderline() {
    let intensity = Array3::<f64>::from_elem((4, 4, 2), 0.5);
    let classification = classify_tissue_types(&intensity);

    for value in classification.iter() {
        assert_eq!(*value, 0.5); // Borderline
    }
}

#[test]
fn test_classify_tissue_types_abnormal() {
    let intensity = Array3::<f64>::from_elem((4, 4, 2), 0.7);
    let classification = classify_tissue_types(&intensity);

    for value in classification.iter() {
        assert_eq!(*value, 1.0); // Moderate abnormality
    }
}

#[test]
fn test_classify_tissue_types_high_abnormal() {
    let intensity = Array3::<f64>::from_elem((4, 4, 2), 0.9);
    let classification = classify_tissue_types(&intensity);

    for value in classification.iter() {
        assert_eq!(*value, 2.0); // High abnormality
    }
}

#[test]
fn test_compute_oxygenation_index_normal() {
    let intensity = Array3::<f64>::from_elem((4, 4, 2), 0.0);
    let oxygenation = compute_oxygenation_index(&intensity);

    for value in oxygenation.iter() {
        assert!((value - 0.75).abs() < 1e-10); // Baseline oxygenation
    }
}

#[test]
fn test_compute_oxygenation_index_high() {
    let intensity = Array3::<f64>::from_elem((4, 4, 2), 1.0);
    let oxygenation = compute_oxygenation_index(&intensity);

    for value in oxygenation.iter() {
        let expected = (0.75_f64 + 1.0 * 0.6 * 0.4).min(1.0);
        assert!((value - expected).abs() < 1e-10);
    }
}

#[test]
fn test_compute_oxygenation_index_clamped() {
    let intensity = Array3::<f64>::from_elem((4, 4, 2), 2.0);
    let oxygenation = compute_oxygenation_index(&intensity);

    for value in oxygenation.iter() {
        assert!(*value <= 1.0); // Should be clamped to 1.0
    }
}

#[test]
fn test_compute_composite_stiffness_low_intensity() {
    let intensity = Array3::<f64>::from_elem((4, 4, 2), 0.0);
    let stiffness = compute_composite_stiffness(&intensity);

    for value in stiffness.iter() {
        let expected = BASELINE_SOFT_TISSUE_STIFFNESS_KPA * (1.0 + 1.0 * STIFFNESS_INTENSITY_COUPLING); // 60 kPa
        assert!((value - expected).abs() < 1e-10);
    }
}

#[test]
fn test_compute_composite_stiffness_high_intensity() {
    let intensity = Array3::<f64>::from_elem((4, 4, 2), 1.0);
    let stiffness = compute_composite_stiffness(&intensity);

    for value in stiffness.iter() {
        let expected = BASELINE_SOFT_TISSUE_STIFFNESS_KPA * (1.0 + 0.0 * STIFFNESS_INTENSITY_COUPLING); // 20 kPa
        assert!((value - expected).abs() < 1e-10);
    }
}

#[test]
fn test_compute_composite_stiffness_range() {
    let mut intensity = Array3::<f64>::zeros((4, 4, 2));
    intensity[[0, 0, 0]] = 0.0;
    intensity[[1, 1, 1]] = 1.0;

    let stiffness = compute_composite_stiffness(&intensity);

    let min_stiffness = stiffness.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_stiffness = stiffness.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    assert!(min_stiffness >= BASELINE_SOFT_TISSUE_STIFFNESS_KPA);
    assert!(max_stiffness <= BASELINE_SOFT_TISSUE_STIFFNESS_KPA * (1.0 + STIFFNESS_INTENSITY_COUPLING));
}

#[test]
fn test_compute_tissue_density() {
    let intensity = Array3::<f64>::from_elem((4, 4, 2), 0.25);
    let density = compute_tissue_density(&intensity);

    for value in density.iter() {
        let expected = 0.25_f64.powf(DENSITY_NONLINEARITY_EXPONENT);
        assert!((value - expected).abs() < 1e-10); // sqrt(0.25) = 0.5
    }
}

#[test]
fn test_detect_regions_of_interest() {
    let mut classification = Array3::<f64>::zeros((4, 4, 2));
    classification[[0, 0, 0]] = 0.0; // Normal
    classification[[1, 1, 1]] = 1.0; // Abnormal
    classification[[2, 2, 0]] = 2.0; // High abnormal

    let roi = detect_regions_of_interest(&classification, 1.0);

    assert_eq!(roi[[0, 0, 0]], 0.0); // Not ROI
    assert_eq!(roi[[1, 1, 1]], 1.0); // ROI
    assert_eq!(roi[[2, 2, 0]], 1.0); // ROI
}

#[test]
fn test_extract_tissue_properties_keys() {
    let fused_result = FusedImageResult {
        intensity_image: Array3::<f64>::zeros((4, 4, 2)),
        tissue_properties: HashMap::new(),
        confidence_map: Array3::<f64>::zeros((4, 4, 2)),
        uncertainty_map: None,
        registration_transforms: HashMap::new(),
        modality_quality: HashMap::new(),
        coordinates: [vec![0.0], vec![0.0], vec![0.0]],
    };

    let properties = extract_tissue_properties(&fused_result);

    assert!(properties.contains_key("tissue_classification"));
    assert!(properties.contains_key("oxygenation_index"));
    assert!(properties.contains_key("composite_stiffness"));
}
