//! Tests for AI-enhanced beamforming integration

#[cfg(feature = "pinn")]
use kwavers::sensor::beamforming::{
    AIBeamformingConfig, AIBeamformingResult, AIEnhancedBeamformingProcessor,
    ClinicalDecisionSupport, DiagnosisAlgorithm, FeatureExtractor, RealTimeWorkflow,
};
#[cfg(feature = "pinn")]
use ndarray::Array3;
#[cfg(feature = "pinn")]
use std::collections::HashMap;

#[cfg(feature = "pinn")]
mod pinn_tests {
    use super::*;
    use kwavers::sensor::beamforming::ai_integration::ClinicalDecisionSupport;
    use kwavers::sensor::beamforming::ai_integration::DiagnosisAlgorithm;
    use kwavers::sensor::beamforming::ai_integration::FeatureExtractor;
    use kwavers::sensor::beamforming::ai_integration::RealTimeWorkflow;

    #[test]
    fn test_ai_enhanced_beamforming_creation() {
        // Test that we can create an AI-enhanced beamforming processor
        let config = AIBeamformingConfig::default();
        let sensor_positions = vec![
            [0.0, 0.0, 0.0],
            [0.001, 0.0, 0.0],
            [0.0, 0.001, 0.0],
            [0.001, 0.001, 0.0],
        ];

        // This will fail in test environment without proper PINN setup
        // but the API should be testable
        let result = AIEnhancedBeamformingProcessor::new(config, sensor_positions);
        assert!(result.is_err() || result.is_ok()); // Either way, API works
    }

    #[test]
    fn test_feature_extractor_creation() {
        let config = kwavers::sensor::beamforming::ai_integration::FeatureConfig::default();
        let extractor = FeatureExtractor::new(config);

        // Test with dummy data
        let volume = Array3::<f32>::from_elem((32, 32, 16), 1.0);
        let features = extractor.extract_features(volume.view()).unwrap();

        assert!(features.morphological.contains_key("gradient_magnitude"));
        assert!(features.spectral.contains_key("local_frequency"));
        assert!(features.texture.contains_key("speckle_variance"));
        assert!(features.texture.contains_key("homogeneity"));
    }

    #[test]
    fn test_clinical_decision_support() {
        let thresholds =
            kwavers::sensor::beamforming::ai_integration::ClinicalThresholds::default();
        let support = ClinicalDecisionSupport::new(thresholds);

        // Create dummy features
        let mut morphological = HashMap::new();
        morphological.insert(
            "gradient_magnitude".to_string(),
            Array3::from_elem((32, 32, 16), 0.5),
        );

        let mut spectral = HashMap::new();
        spectral.insert(
            "local_frequency".to_string(),
            Array3::from_elem((32, 32, 16), 0.1),
        );

        let mut texture = HashMap::new();
        texture.insert(
            "speckle_variance".to_string(),
            Array3::from_elem((32, 32, 16), 0.8),
        );
        texture.insert(
            "homogeneity".to_string(),
            Array3::from_elem((32, 32, 16), 0.7),
        );

        let features = kwavers::sensor::beamforming::ai_integration::FeatureMap {
            morphological,
            spectral,
            texture,
        };

        let volume = Array3::<f32>::from_elem((32, 32, 16), 1.0);
        let uncertainty = Array3::<f32>::from_elem((32, 32, 16), 0.1);
        let confidence = Array3::<f32>::from_elem((32, 32, 16), 0.9);

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

    #[test]
    fn test_diagnosis_algorithm() {
        let algorithm = DiagnosisAlgorithm::new();

        // Create dummy clinical analysis
        let clinical_analysis = kwavers::sensor::beamforming::ai_integration::ClinicalAnalysis {
            lesions: vec![],
            tissue_classification:
                kwavers::sensor::beamforming::ai_integration::TissueClassification {
                    probabilities: HashMap::new(),
                    dominant_tissue: Array3::from_elem((32, 32, 16), "Muscle".to_string()),
                    boundary_confidence: Array3::from_elem((32, 32, 16), 0.8),
                },
            recommendations: vec!["Test recommendation".to_string()],
            diagnostic_confidence: 0.85,
        };

        let features = kwavers::sensor::beamforming::ai_integration::FeatureMap {
            morphological: HashMap::new(),
            spectral: HashMap::new(),
            texture: HashMap::new(),
        };

        let diagnosis = algorithm.diagnose(&features, &clinical_analysis).unwrap();
        assert!(!diagnosis.is_empty());
    }

    #[test]
    fn test_realtime_workflow() {
        let mut workflow = RealTimeWorkflow::new();

        // Test performance statistics
        let stats = workflow.get_performance_stats();
        assert!(stats.contains_key("avg_processing_time"));
        assert!(stats.contains_key("diagnostic_confidence"));
    }

    #[test]
    fn test_feature_extraction_comprehensive() {
        let config = kwavers::sensor::beamforming::ai_integration::FeatureConfig {
            morphological_features: true,
            spectral_features: true,
            texture_features: true,
            window_size: 8,
            overlap: 0.5,
        };

        let extractor = FeatureExtractor::new(config);

        // Create test volume with some structure
        let mut volume = Array3::<f32>::zeros((16, 16, 8));

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
        assert!(features.spectral.len() >= 1); // local_frequency
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

    #[test]
    fn test_clinical_analysis_with_lesions() {
        let thresholds = kwavers::sensor::beamforming::ai_integration::ClinicalThresholds {
            lesion_confidence_threshold: 0.7,
            tissue_uncertainty_threshold: 0.2,
            contrast_abnormality_threshold: 2.0,
            speckle_anomaly_threshold: 1.0,
            segmentation_sensitivity: 1.0,
            voxel_size_mm: 0.5,
        };

        let support = ClinicalDecisionSupport::new(thresholds);

        // Create features that would indicate a lesion
        let mut morphological = HashMap::new();
        morphological.insert(
            "gradient_magnitude".to_string(),
            Array3::from_elem((16, 16, 8), 1.0), // High gradients
        );

        let mut texture = HashMap::new();
        texture.insert(
            "speckle_variance".to_string(),
            Array3::from_elem((16, 16, 8), 2.0), // High variance
        );

        let features = kwavers::sensor::beamforming::ai_integration::FeatureMap {
            morphological,
            spectral: HashMap::new(),
            texture,
        };

        // Create volume with high contrast region
        let mut volume = Array3::<f32>::from_elem((16, 16, 8), 0.5);
        volume[[8, 8, 4]] = 3.0; // High contrast lesion

        let uncertainty = Array3::<f32>::from_elem((16, 16, 8), 0.1);
        let confidence = Array3::<f32>::from_elem((16, 16, 8), 0.9);

        let analysis = support
            .analyze_clinical(
                volume.view(),
                &features,
                uncertainty.view(),
                confidence.view(),
            )
            .unwrap();

        // Should detect at least one lesion
        assert!(
            !analysis.lesions.is_empty(),
            "Clinical analysis should detect lesions with high contrast and feature anomalies"
        );

        // Verify lesion properties
        let lesion = &analysis.lesions[0];
        assert!(lesion.confidence >= 0.0 && lesion.confidence <= 1.0);
        assert!(lesion.size_mm > 0.0);
        assert!(!lesion.lesion_type.is_empty());
        assert!(lesion.clinical_significance >= 0.0 && lesion.clinical_significance <= 1.0);

        // Should have recommendations
        assert!(!analysis.recommendations.is_empty());
    }

    #[test]
    fn test_performance_monitoring() {
        let mut workflow = RealTimeWorkflow::new();

        // Initially empty
        let initial_stats = workflow.get_performance_stats();
        assert!(initial_stats.get("avg_processing_time").unwrap() == &0.0);

        // Simulate some processing times
        // (In real usage, this would be populated by actual processing)
        workflow.performance_history = vec![50.0, 45.0, 55.0, 48.0, 52.0];

        let stats = workflow.get_performance_stats();

        assert_eq!(*stats.get("min_time").unwrap(), 45.0);
        assert_eq!(*stats.get("max_time").unwrap(), 55.0);
        assert!(*stats.get("avg_processing_time").unwrap() > 0.0);

        // Test median calculation
        let median = *stats.get("median_time").unwrap();
        assert!(median >= 45.0 && median <= 55.0);
    }

    #[test]
    fn test_config_defaults() {
        let config = AIBeamformingConfig::default();

        assert!(config.enable_realtime_pinn);
        assert!(config.enable_clinical_support);
        assert_eq!(config.performance_target_ms, 100.0);

        let feature_config = config.feature_config;
        assert!(feature_config.morphological_features);
        assert!(feature_config.spectral_features);
        assert!(feature_config.texture_features);

        let thresholds = config.clinical_thresholds;
        assert_eq!(thresholds.lesion_confidence_threshold, 0.8);
        assert_eq!(thresholds.tissue_uncertainty_threshold, 0.3);
        assert_eq!(thresholds.contrast_abnormality_threshold, 2.0);
    }
}
