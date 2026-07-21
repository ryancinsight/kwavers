use super::extractor::FeatureExtractor;
use eunomia::assert_relative_eq;
use kwavers_analysis::signal_processing::beamforming::neural::config::FeatureConfig;
use leto::Array3;

#[test]
fn test_feature_extractor_creation() {
    let config = FeatureConfig::default();
    let extractor = FeatureExtractor::new(config);
    assert!(extractor.config.morphological_features);
}

#[test]
fn test_extract_all_features() {
    let config = FeatureConfig::default();
    let extractor = FeatureExtractor::new(config);

    let volume = Array3::<f32>::from_elem((10, 10, 10), 1.0);
    let features = extractor.extract_features(volume.view()).unwrap();

    assert!(features.morphological.contains_key("gradient_magnitude"));
    assert!(features.morphological.contains_key("laplacian"));
    assert!(features.spectral.contains_key("local_frequency"));
    assert!(features.texture.contains_key("speckle_variance"));
    assert!(features.texture.contains_key("homogeneity"));
}

#[test]
fn test_selective_feature_extraction() {
    let config = FeatureConfig {
        morphological_features: true,
        spectral_features: false,
        texture_features: false,
        ..Default::default()
    };

    let extractor = FeatureExtractor::new(config);
    let volume = Array3::<f32>::from_elem((10, 10, 10), 1.0);
    let features = extractor.extract_features(volume.view()).unwrap();

    assert!(features.morphological.contains_key("gradient_magnitude"));
    assert!(features.spectral.is_empty());
    assert!(features.texture.is_empty());
}

#[test]
fn test_gradient_magnitude_constant_volume() {
    let config = FeatureConfig::default();
    let extractor = FeatureExtractor::new(config);

    // Constant volume should have zero gradient everywhere (except boundaries)
    let volume = Array3::<f32>::from_elem((10, 10, 10), 5.0);
    let gradient = extractor.compute_gradient_magnitude(volume.view());

    // Check interior points (boundaries may have artifacts)
    for z in 2..8 {
        for y in 2..8 {
            for x in 2..8 {
                assert_relative_eq!(gradient[[x, y, z]], 0.0, epsilon = 1e-6);
            }
        }
    }
}

#[test]
fn test_gradient_magnitude_step_edge() {
    let config = FeatureConfig::default();
    let extractor = FeatureExtractor::new(config);

    // Create volume with step edge in x-direction
    let mut volume = Array3::<f32>::zeros((10, 10, 10));
    for z in 0..10 {
        for y in 0..10 {
            for x in 5..10 {
                volume[[x, y, z]] = 1.0;
            }
        }
    }

    let gradient = extractor.compute_gradient_magnitude(volume.view());

    // Gradient should be strong at x=5 (edge location)
    // Central difference at x=5: (1.0 - 0.0) / 2.0 = 0.5
    assert!(gradient[[5, 5, 5]] > 0.4);
    assert!(gradient[[5, 5, 5]] < 0.6);
}

#[test]
fn test_laplacian_constant_volume() {
    let config = FeatureConfig::default();
    let extractor = FeatureExtractor::new(config);

    // Laplacian of constant function is zero
    let volume = Array3::<f32>::from_elem((10, 10, 10), 3.0);
    let laplacian = extractor.compute_laplacian(volume.view());

    for z in 2..8 {
        for y in 2..8 {
            for x in 2..8 {
                assert_relative_eq!(laplacian[[x, y, z]], 0.0, epsilon = 1e-6);
            }
        }
    }
}

#[test]
fn test_laplacian_spherical_blob() {
    let config = FeatureConfig::default();
    let extractor = FeatureExtractor::new(config);

    // Create spherical blob
    let mut volume = Array3::<f32>::zeros((20, 20, 20));
    let center = (10.0, 10.0, 10.0);
    let radius = 5.0;

    for z in 0..20 {
        for y in 0..20 {
            for x in 0..20 {
                let dist = ((x as f32 - center.0).powi(2)
                    + (y as f32 - center.1).powi(2)
                    + (z as f32 - center.2).powi(2))
                .sqrt();
                if dist < radius {
                    volume[[x, y, z]] = 1.0;
                }
            }
        }
    }

    let laplacian = extractor.compute_laplacian(volume.view());

    assert_relative_eq!(laplacian[[10, 10, 10]], 0.0, epsilon = 1e-6);
    assert!(laplacian[[14, 10, 10]] < 0.0);
    assert!(laplacian[[15, 10, 10]] > 0.0);
}

#[test]
fn test_speckle_variance_uniform_region() {
    let config = FeatureConfig::default();
    let extractor = FeatureExtractor::new(config);

    // Uniform region should have low variance
    let volume = Array3::<f32>::from_elem((40, 40, 40), 1.0);
    let variance = extractor.compute_speckle_variance(volume.view());

    // Check central region
    let center_variance = variance[[20, 20, 20]];
    assert_relative_eq!(center_variance, 0.0, epsilon = 1e-6);
}

#[test]
fn test_homogeneity_uniform_region() {
    let config = FeatureConfig::default();
    let extractor = FeatureExtractor::new(config);

    // Uniform region should have maximum homogeneity (1.0)
    let volume = Array3::<f32>::from_elem((10, 10, 10), 2.0);
    let homogeneity = extractor.compute_homogeneity(volume.view());

    // Homogeneity = 1 / (1 + 0) = 1.0 for uniform region
    assert_relative_eq!(homogeneity[[5, 5, 5]], 1.0, epsilon = 1e-6);
}

#[test]
fn test_local_frequency_constant_region() {
    let config = FeatureConfig::default();
    let extractor = FeatureExtractor::new(config);

    // Constant region has zero variance (low frequency)
    let volume = Array3::<f32>::from_elem((10, 10, 10), 1.0);
    let frequency = extractor.compute_local_frequency(volume.view());

    assert_relative_eq!(frequency[[5, 5, 5]], 0.0, epsilon = 1e-6);
}

#[test]
fn test_feature_extraction_preserves_dimensions() {
    let config = FeatureConfig::default();
    let extractor = FeatureExtractor::new(config);

    let volume = Array3::<f32>::zeros((20, 30, 40));
    let features = extractor.extract_features(volume.view()).unwrap();

    // All features should have same dimensions as input
    for (_, feature_array) in features.morphological.iter() {
        assert_eq!(feature_array.shape(), [20, 30, 40]);
    }
    for (_, feature_array) in features.spectral.iter() {
        assert_eq!(feature_array.shape(), [20, 30, 40]);
    }
    for (_, feature_array) in features.texture.iter() {
        assert_eq!(feature_array.shape(), [20, 30, 40]);
    }
}
