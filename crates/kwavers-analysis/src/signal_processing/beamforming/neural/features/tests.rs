use super::computation::{
    compute_laplacian, compute_local_entropy, compute_local_std, compute_spatial_gradient,
    concatenate_features, extract_all_features, normalize_features,
};
use leto::Array3;

fn create_test_image() -> Array3<f32> {
    let mut image = Array3::zeros((1, 10, 10));
    // Create a simple pattern: bright center, dark edges
    for i in 3..7 {
        for j in 3..7 {
            image[[0, i, j]] = 1.0;
        }
    }
    image
}

#[test]
fn test_extract_all_features() {
    let image = create_test_image();
    let features = extract_all_features(&image);
    // Should return 6 summary statistics: mean, std, gradient, laplacian, entropy, peak
    assert_eq!(features.len(), 6);
    // All features should be non-negative (or reasonable values)
    for &feature_val in features.iter() {
        assert!(feature_val.is_finite());
    }
}

#[test]
fn test_local_std() {
    let image = create_test_image();
    let std_map = compute_local_std(&image);
    assert_eq!(std_map.shape(), [1, 10, 10]);
    // Borders should be zero
    assert_eq!(std_map[[0, 0, 0]], 0.0);
    // Center should have some variation at edges
    assert!(std_map[[0, 4, 4]] >= 0.0);
}

#[test]
fn test_spatial_gradient() {
    let image = create_test_image();
    let grad_map = compute_spatial_gradient(&image);
    assert_eq!(grad_map.shape(), [1, 10, 10]);
    // Gradient should be high at edges of bright region
    assert!(grad_map[[0, 3, 3]] > 0.0 || grad_map[[0, 3, 4]] > 0.0);
}

#[test]
fn test_laplacian() {
    let image = create_test_image();
    let lap_map = compute_laplacian(&image);
    assert_eq!(lap_map.shape(), [1, 10, 10]);
    // Laplacian should detect edges
    assert!(lap_map[[0, 5, 5]] >= 0.0);
}

#[test]
fn test_local_entropy() {
    let image = create_test_image();
    let entropy_map = compute_local_entropy(&image);
    assert_eq!(entropy_map.shape(), [1, 10, 10]);
    // Entropy should be higher at edges (more variation)
    assert!(entropy_map[[0, 3, 3]] >= 0.0);
}

#[test]
fn test_normalize_features() {
    // Create multiple feature maps for normalization testing
    let image = create_test_image();
    let mut features = vec![
        image.clone(),
        compute_local_std(&image),
        compute_spatial_gradient(&image),
    ];
    normalize_features(&mut features);

    for feature in &features {
        let min_val = feature.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = feature.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(min_val >= -1e-6); // Allow small numerical error
        assert!(max_val <= 1.0 + 1e-6);
    }
}

#[test]
fn test_normalize_features_value_semantics() {
    let mut features = vec![Array3::from_shape_vec((1, 1, 4), vec![2.0, 4.0, 6.0, 10.0]).unwrap()];

    normalize_features(&mut features);

    let expected = Array3::from_shape_vec((1, 1, 4), vec![0.0, 0.25, 0.5, 1.0]).unwrap();
    assert_eq!(features[0], expected);
}

#[test]
fn test_concatenate_features() {
    // Create multiple feature maps for concatenation testing
    let image = create_test_image();
    let features = vec![
        image.clone(),
        compute_local_std(&image),
        compute_spatial_gradient(&image),
    ];
    let stacked = concatenate_features(&features);
    assert_eq!(stacked.shape(), [1, 3, 10, 10]);
}

#[test]
fn test_concatenate_empty() {
    let features: Vec<Array3<f32>> = vec![];
    let stacked = concatenate_features(&features);
    assert_eq!(stacked.shape(), [0, 0, 0, 0]);
}

#[test]
fn test_edge_cases_uniform_image() {
    let image = Array3::from_elem((1, 10, 10), 0.5);
    let features = extract_all_features(&image);

    // Uniform image should have:
    // - Mean = 0.5
    // - Zero std/gradient/laplacian
    // - Low entropy
    // - Peak = 0.5
    assert!((features[0] - 0.5).abs() < 1e-6); // mean
    assert!(features[1].abs() < 1e-6); // std
    assert!(features[2].abs() < 1e-6); // gradient
    assert!(features[3].abs() < 1e-6); // laplacian
                                       // Entropy can be > 0 for uniform (depends on binning)
    assert!((features[5] - 0.5).abs() < 1e-6); // peak
}

#[test]
fn test_zero_image() {
    let image = Array3::zeros((1, 10, 10));
    let features = extract_all_features(&image);
    assert_eq!(features.len(), 6);
    // All features should be zero for zero image
    for &feature_val in features.iter() {
        assert!(feature_val.abs() < 1e-6);
    }
}
