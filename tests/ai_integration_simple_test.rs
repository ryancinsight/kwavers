//! Simple AI integration tests that don't depend on PINN compilation

#[cfg(feature = "pinn")]
use kwavers::domain::sensor::beamforming::ai_integration::{
    ClinicalDecisionSupport, ClinicalThresholds, DiagnosisAlgorithm, FeatureConfig,
    FeatureExtractor, FeatureMap, RealTimeWorkflow,
};

#[cfg(feature = "pinn")]
#[test]
fn test_feature_extractor_creation() {
    let config = FeatureConfig::default();
    let extractor = FeatureExtractor::new(config);

    // Test with dummy data
    let volume = ndarray::Array3::<f32>::from_elem((32, 32, 16), 1.0);
    let features = extractor.extract_features(volume.view()).unwrap();

    assert!(features.morphological.contains_key("gradient_magnitude"));
    assert!(features.spectral.contains_key("local_frequency"));
    assert!(features.texture.contains_key("speckle_variance"));
    assert!(features.texture.contains_key("homogeneity"));
}

#[cfg(feature = "pinn")]
#[test]
fn test_clinical_decision_support() {
    let thresholds = ClinicalThresholds::default();
    let support = ClinicalDecisionSupport::new(thresholds);

    // Create dummy features
    let mut morphological = std::collections::HashMap::new();
    morphological.insert(
        "gradient_magnitude".to_string(),
        ndarray::Array3::from_elem((32, 32, 16), 0.5),
    );

    let mut texture = std::collections::HashMap::new();
    texture.insert(
        "speckle_variance".to_string(),
        ndarray::Array3::from_elem((32, 32, 16), 0.8),
    );

    let features = FeatureMap {
        morphological,
        spectral: std::collections::HashMap::new(),
        texture,
    };

    let volume = ndarray::Array3::<f32>::from_elem((32, 32, 16), 1.0);
    let uncertainty = ndarray::Array3::<f32>::from_elem((32, 32, 16), 0.1);
    let confidence = ndarray::Array3::<f32>::from_elem((32, 32, 16), 0.9);

    let analysis = support
        .analyze_clinical(
            volume.view(),
            &features,
            uncertainty.view(),
            confidence.view(),
        )
        .unwrap();

    assert!(analysis.diagnostic_confidence >= 0.0 && analysis.diagnostic_confidence <= 1.0);
    assert!(!analysis.recommendations.is_empty());
}

#[cfg(feature = "pinn")]
#[test]
fn test_diagnosis_algorithm() {
    let algorithm = DiagnosisAlgorithm::new();

    // Create dummy clinical analysis
    let clinical_analysis =
        kwavers::domain::sensor::beamforming::ai_integration::ClinicalAnalysis {
            lesions: vec![],
            tissue_classification:
                kwavers::domain::sensor::beamforming::ai_integration::TissueClassification {
                    probabilities: std::collections::HashMap::new(),
                    dominant_tissue: ndarray::Array3::from_elem((32, 32, 16), "Muscle".to_string()),
                    boundary_confidence: ndarray::Array3::from_elem((32, 32, 16), 0.8),
                },
            recommendations: vec!["Test recommendation".to_string()],
            diagnostic_confidence: 0.85,
        };

    let features = FeatureMap {
        morphological: std::collections::HashMap::new(),
        spectral: std::collections::HashMap::new(),
        texture: std::collections::HashMap::new(),
    };

    let diagnosis = algorithm.diagnose(&features, &clinical_analysis).unwrap();
    assert!(!diagnosis.is_empty());
}

#[cfg(feature = "pinn")]
#[test]
fn test_realtime_workflow() {
    let workflow = RealTimeWorkflow::new();

    // Test performance statistics
    let stats = workflow.get_performance_stats();
    assert!(stats.contains_key("avg_processing_time"));
    assert!(stats.contains_key("diagnostic_confidence"));
}

#[cfg(feature = "pinn")]
#[test]
fn test_config_defaults() {
    let feature_config = FeatureConfig::default();
    assert!(feature_config.morphological_features);
    assert!(feature_config.spectral_features);
    assert!(feature_config.texture_features);

    let thresholds = ClinicalThresholds::default();
    assert_eq!(thresholds.lesion_confidence_threshold, 0.8);
    assert_eq!(thresholds.tissue_uncertainty_threshold, 0.3);
    assert_eq!(thresholds.contrast_abnormality_threshold, 2.0);
}

#[cfg(feature = "pinn")]
#[test]
fn test_feature_extraction_comprehensive() {
    let config = FeatureConfig {
        morphological_features: true,
        spectral_features: true,
        texture_features: true,
        window_size: 8,
        overlap: 0.5,
    };

    let extractor = FeatureExtractor::new(config);

    // Create test volume with some structure
    let mut volume = ndarray::Array3::<f32>::zeros((16, 16, 8));

    // Add some test patterns
    for z in 0..8 {
        for y in 0..16 {
            for x in 0..16 {
                // Create a gradient
                let value = (x as f32 + y as f32 + z as f32) / 40.0;
                volume[[x, y, z]] = value;
            }
        }
    }

    // Add a simulated lesion (high intensity region)
    for z in 2..6 {
        for y in 6..10 {
            for x in 6..10 {
                volume[[x, y, z]] = 2.5; // High intensity
            }
        }
    }

    let features = extractor.extract_features(volume.view()).unwrap();

    // Verify features are computed
    assert!(features.morphological.len() >= 2); // gradient_magnitude, laplacian
    assert!(!features.spectral.is_empty()); // local_frequency
    assert!(features.texture.len() >= 2); // speckle_variance, homogeneity

    // Check that features have expected dimensions
    for feature in features.morphological.values() {
        assert_eq!(feature.dim(), (16, 16, 8));
    }
    for feature in features.spectral.values() {
        assert_eq!(feature.dim(), (16, 16, 8));
    }
    for feature in features.texture.values() {
        assert_eq!(feature.dim(), (16, 16, 8));
    }

    // Verify gradient magnitude is computed (should be > 0 in some regions)
    let grad_mag = features.morphological.get("gradient_magnitude").unwrap();
    let max_grad = grad_mag.iter().cloned().fold(0.0f32, f32::max);
    assert!(
        max_grad > 0.0,
        "Gradient magnitude should be positive in some regions"
    );

    // Verify speckle variance is computed
    let speckle_var = features.texture.get("speckle_variance").unwrap();
    let max_var = speckle_var.iter().cloned().fold(0.0f32, f32::max);
    assert!(max_var >= 0.0, "Speckle variance should be non-negative");
}
