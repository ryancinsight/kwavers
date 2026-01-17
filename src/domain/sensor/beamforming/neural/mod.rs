//! Neural Beamforming with Real-Time PINN Inference
//!
//! This module integrates real-time Physics-Informed Neural Network (PINN) inference
//! with ultrasound beamforming for clinical decision support and automated diagnosis.
//! Combines traditional signal processing with neural network-enhanced analysis for
//! point-of-care applications.
//!
//! # Architecture
//!
//! ```text
//! Raw RF Data → Beamforming → Feature Extraction → PINN Inference → Clinical Analysis
//!      ↓              ↓               ↓                    ↓                ↓
//!   Array4<f32>   Volume 3D    Morphological/       Uncertainty      Lesion Detection
//!   (time×chan×               Spectral/Texture      Quantification   Tissue Classification
//!    frames×samps)            Features                                Recommendations
//! ```
//!
//! # Module Organization
//!
//! - [`config`]: Configuration types for neural beamforming
//! - [`types`]: Result types, feature maps, and clinical analysis structures
//! - [`processor`]: Main neural beamforming processor (requires `pinn` feature)
//! - [`features`]: Feature extraction algorithms (morphological, spectral, texture)
//! - [`clinical`]: Clinical decision support (lesion detection, tissue classification)
//! - [`diagnosis`]: Automated diagnosis algorithm
//! - [`workflow`]: Real-time workflow manager with performance monitoring
//!
//! # Clinical Applications
//!
//! - Real-time tissue characterization
//! - Automated lesion detection with confidence scoring
//! - Uncertainty-guided imaging protocols
//! - Point-of-care diagnostic assistance
//! - Clinical decision support with recommendations
//!
//! # Performance Requirements
//!
//! Target total processing time: **<100ms** for real-time clinical use
//! - Beamforming: <30ms
//! - Feature extraction: <20ms
//! - PINN inference: <30ms
//! - Clinical analysis: <20ms
//!
//! # Literature References
//!
//! - Raissi et al. (2019): "Physics-informed neural networks: A deep learning framework"
//! - Van Veen & Buckley (1988): "Beamforming: A versatile approach to spatial filtering"
//! - Kendall & Gal (2017): "What uncertainties do we need in Bayesian deep learning?"
//! - Stavros et al. (1995): "Solid breast nodules: use of sonography to distinguish"
//!
//! # Example Usage
//!
//! ```ignore
//! use kwavers::domain::sensor::beamforming::neural::{
//!     config::AIBeamformingConfig,
//!     processor::AIEnhancedBeamformingProcessor,
//!     workflow::RealTimeWorkflow,
//! };
//! use ndarray::Array4;
//!
//! // Configure neural beamforming
//! let config = AIBeamformingConfig::default();
//! let sensor_positions = vec![[0.0, 0.0, 0.0]; 64]; // 64-element array
//!
//! // Create processor and workflow manager
//! let mut processor = AIEnhancedBeamformingProcessor::new(config, sensor_positions)?;
//! let mut workflow = RealTimeWorkflow::new();
//!
//! // Process ultrasound data
//! let rf_data = Array4::<f32>::zeros((1024, 64, 100, 1));
//! let angles = vec![0.0; 100]; // Steering angles
//!
//! let result = workflow.execute_workflow(&mut processor, rf_data.view(), &angles)?;
//!
//! // Access results
//! println!("Detected {} lesions", result.clinical_analysis.lesions.len());
//! println!("Diagnostic confidence: {:.2}", result.clinical_analysis.diagnostic_confidence);
//! println!("Processing time: {:.2}ms", result.performance.total_time_ms);
//!
//! // Monitor performance
//! let stats = workflow.get_performance_stats();
//! println!("Average time: {:.2}ms", stats["avg_processing_time"]);
//! ```
//!
//! # Clinical Safety Notice
//!
//! All neural network analysis results are for **decision support only** and require
//! clinical interpretation by qualified medical professionals. This system does not
//! replace clinical judgment and must be used in accordance with institutional protocols
//! and regulatory requirements.

// Module declarations
pub mod clinical;
pub mod config;
pub mod diagnosis;
pub mod features;
pub mod types;
pub mod workflow;

// Processor module requires PINN feature
#[cfg(feature = "pinn")]
pub mod processor;

// Public re-exports for convenient access
pub use clinical::ClinicalDecisionSupport;
pub use config::{AIBeamformingConfig, ClinicalThresholds, FeatureConfig};
pub use diagnosis::DiagnosisAlgorithm;
pub use features::FeatureExtractor;
pub use types::{
    AIBeamformingResult, ClinicalAnalysis, FeatureMap, LesionDetection, PerformanceMetrics,
    TissueClassification,
};
pub use workflow::RealTimeWorkflow;

// Conditionally export processor when PINN feature is enabled
#[cfg(feature = "pinn")]
pub use processor::AIEnhancedBeamformingProcessor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = AIBeamformingConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.performance_target_ms, 100.0);
    }

    #[test]
    fn test_feature_config_validation() {
        let mut config = FeatureConfig::default();
        assert!(config.validate().is_ok());

        config.window_size = 2; // Invalid: must be odd
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_clinical_thresholds_presets() {
        let high_sens = ClinicalThresholds::high_sensitivity();
        assert!(high_sens.validate().is_ok());
        assert!(high_sens.lesion_confidence_threshold < 0.8);

        let high_spec = ClinicalThresholds::high_specificity();
        assert!(high_spec.validate().is_ok());
        assert!(high_spec.lesion_confidence_threshold > 0.8);
    }

    #[test]
    fn test_feature_extractor_creation() {
        let config = FeatureConfig::default();
        let extractor = FeatureExtractor::new(config);
        let volume = ndarray::Array3::<f32>::zeros((5, 5, 5));
        let features = extractor.extract_features(volume.view()).unwrap();
        assert!(!features.is_empty());
    }

    #[test]
    fn test_clinical_decision_support_creation() {
        let thresholds = ClinicalThresholds::default();
        let support = ClinicalDecisionSupport::new(thresholds);
        let volume = ndarray::Array3::<f32>::zeros((5, 5, 5));
        let features = FeatureMap::new();
        let uncertainty = ndarray::Array3::<f32>::zeros((5, 5, 5));
        let confidence = ndarray::Array3::<f32>::ones((5, 5, 5));
        let analysis = support
            .analyze_clinical(
                volume.view(),
                &features,
                uncertainty.view(),
                confidence.view(),
            )
            .unwrap();
        assert!(!analysis.has_lesions());
        assert!((0.0..=1.0).contains(&analysis.diagnostic_confidence));
    }

    #[test]
    fn test_diagnosis_algorithm_creation() {
        let algorithm = DiagnosisAlgorithm::new();
        let default_algorithm = DiagnosisAlgorithm::default();
        let features = FeatureMap::new();
        let analysis = ClinicalAnalysis::empty();
        let diagnosis_1 = algorithm.diagnose(&features, &analysis).unwrap();
        let diagnosis_2 = default_algorithm.diagnose(&features, &analysis).unwrap();
        assert!(!diagnosis_1.is_empty());
        assert!(!diagnosis_2.is_empty());
    }

    #[test]
    fn test_workflow_creation() {
        let workflow = RealTimeWorkflow::new();
        assert!(workflow.performance_history.is_empty());
        assert!(!workflow.meets_performance_target());
    }

    #[test]
    fn test_feature_map_operations() {
        let mut features = FeatureMap::new();
        assert!(features.is_empty());
        assert_eq!(features.feature_count(), 0);

        features
            .morphological
            .insert("gradient".to_string(), ndarray::Array3::zeros((10, 10, 10)));
        assert!(!features.is_empty());
        assert_eq!(features.feature_count(), 1);
    }

    #[test]
    fn test_clinical_analysis_queries() {
        use types::LesionDetection;

        let mut analysis = ClinicalAnalysis::empty();
        assert!(!analysis.has_lesions());
        assert_eq!(analysis.high_confidence_lesion_count(), 0);

        analysis.lesions.push(LesionDetection {
            center: (10, 10, 10),
            size_mm: 5.0,
            confidence: 0.95,
            lesion_type: "Solid".to_string(),
            clinical_significance: 0.85,
        });

        assert!(analysis.has_lesions());
        assert_eq!(analysis.high_confidence_lesion_count(), 1);
    }
}
