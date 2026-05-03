//! Registration, coordinate, quality, and affine-transform tests.

use super::super::*;
use ndarray::Array3;

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
