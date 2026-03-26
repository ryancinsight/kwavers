use super::coordinates::generate_coordinate_arrays;
use super::resampling::{resample_to_target_grid, trilinear_interpolate};
use super::transforms::apply_inverse_transform;
use super::validation::validate_registration_compatibility;
use ndarray::Array3;

#[test]
fn test_generate_coordinate_arrays() {
    let dims = (16, 8, 4);
    let resolution = [1e-4, 2e-4, 3e-4];
    let coords = generate_coordinate_arrays(dims, resolution);

    assert_eq!(coords[0].len(), dims.0);
    assert_eq!(coords[1].len(), dims.1);
    assert_eq!(coords[2].len(), dims.2);

    assert_eq!(coords[0][0], 0.0);
    assert_eq!(coords[1][0], 0.0);
    assert_eq!(coords[2][0], 0.0);

    assert!((coords[0][1] - resolution[0]).abs() < f64::EPSILON);
    assert!((coords[1][1] - resolution[1]).abs() < f64::EPSILON);
    assert!((coords[2][1] - resolution[2]).abs() < f64::EPSILON);
}

#[test]
fn test_trilinear_interpolation_at_grid_point() {
    let image = Array3::<f64>::from_shape_fn((4, 4, 4), |(i, j, k)| (i + j + k) as f64);

    // Test at exact grid point
    let value = trilinear_interpolate(&image, [1.0, 1.0, 1.0], image.shape());
    assert!((value - 3.0).abs() < 1e-10);
}

#[test]
fn test_trilinear_interpolation_midpoint() {
    let image = Array3::<f64>::from_shape_fn((4, 4, 4), |(i, j, k)| (i + j + k) as f64);

    // Test at midpoint between grid points
    let value = trilinear_interpolate(&image, [1.5, 1.5, 1.5], image.shape());
    let expected = (3.0 + 4.0 + 4.0 + 5.0 + 4.0 + 5.0 + 5.0 + 6.0) / 8.0;
    assert!((value - expected).abs() < 1e-10);
}

#[test]
fn test_trilinear_interpolation_clamping() {
    let image = Array3::<f64>::ones((4, 4, 4));

    // Test coordinates outside valid range
    let value = trilinear_interpolate(&image, [-1.0, -1.0, -1.0], image.shape());
    assert!((value - 1.0).abs() < 1e-10);

    let value = trilinear_interpolate(&image, [10.0, 10.0, 10.0], image.shape());
    assert!((value - 1.0).abs() < 1e-10);
}

#[test]
fn test_apply_inverse_transform_identity() {
    let identity = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    let point = [1.0, 2.0, 3.0];
    let transformed = apply_inverse_transform(&identity, point);

    assert!((transformed[0] - point[0]).abs() < 1e-10);
    assert!((transformed[1] - point[1]).abs() < 1e-10);
    assert!((transformed[2] - point[2]).abs() < 1e-10);
}

#[test]
fn test_apply_inverse_transform_translation() {
    let translation = [
        1.0, 0.0, 0.0, 5.0, 0.0, 1.0, 0.0, 10.0, 0.0, 0.0, 1.0, 15.0, 0.0, 0.0, 0.0, 1.0,
    ];

    let point = [5.0, 10.0, 15.0];
    let transformed = apply_inverse_transform(&translation, point);

    assert!((transformed[0] - 0.0).abs() < 1e-10);
    assert!((transformed[1] - 0.0).abs() < 1e-10);
    assert!((transformed[2] - 0.0).abs() < 1e-10);
}

#[test]
fn test_resample_identity_transform() {
    let source = Array3::<f64>::from_elem((4, 4, 4), 1.0);
    let identity = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    let resampled = resample_to_target_grid(&source, &identity, (4, 4, 4));

    for value in resampled.iter() {
        assert!((value - 1.0).abs() < 1e-10);
    }
}

#[test]
fn test_validate_registration_compatibility_valid() {
    let result = validate_registration_compatibility((10, 10, 10), (20, 20, 20));
    assert!(result.is_ok());
}

#[test]
fn test_validate_registration_compatibility_invalid() {
    let result = validate_registration_compatibility((10, 10, 10), (200, 200, 200));
    assert!(result.is_err());
}
