//! Configuration types for multi-modal imaging fusion
//!
//! Note: FusionConfig and FusionMethod have been moved to domain::imaging::fusion
//! for clean architecture compliance. This module now only contains physics-specific
//! registration configuration..
//!
//! This module defines the configuration structures and enums that control
//! how different imaging modalities are combined, including fusion methods,
//! registration strategies, and quality parameters.

use std::collections::HashMap;

/// Multi-modal fusion configuration
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Spatial resolution for fusion output (m)
    pub output_resolution: [f64; 3],
    /// Fusion method to use
    pub fusion_method: FusionMethod,
    /// Registration method for image alignment
    pub registration_method: RegistrationMethod,
    /// Weight factors for each modality
    pub modality_weights: HashMap<String, f64>,
    /// Confidence threshold for fusion decisions
    pub confidence_threshold: f64,
    /// Enable uncertainty quantification
    pub uncertainty_quantification: bool,
}

impl Default for FusionConfig {
    fn default() -> Self {
        let mut modality_weights = HashMap::new();
        modality_weights.insert("ultrasound".to_string(), 0.4);
        modality_weights.insert("photoacoustic".to_string(), 0.35);
        modality_weights.insert("elastography".to_string(), 0.25);

        Self {
            output_resolution: [1e-4, 1e-4, 1e-4], // 100μm isotropic
            fusion_method: FusionMethod::WeightedAverage,
            registration_method: RegistrationMethod::RigidBody,
            modality_weights,
            confidence_threshold: 0.7,
            uncertainty_quantification: true,
        }
    }
}

/// Fusion methods for combining multi-modal data
///
/// Different fusion methods are appropriate for different clinical scenarios
/// and data quality conditions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FusionMethod {
    /// Simple weighted average of modalities
    ///
    /// Fast and robust, suitable for well-registered data with similar
    /// quality across modalities.
    WeightedAverage,
    /// Feature-based fusion using tissue properties
    ///
    /// Extracts and combines complementary tissue features from each modality.
    FeatureBased,
    /// Probabilistic fusion with uncertainty modeling
    ///
    /// Bayesian approach that accounts for uncertainty in each modality.
    Probabilistic,
    /// Deep learning-based fusion
    ///
    /// Neural network-based fusion for complex multi-modal relationships.
    DeepFusion,
    /// Maximum likelihood estimation
    ///
    /// Statistical estimation method that maximizes the likelihood of the
    /// observed multi-modal data.
    MaximumLikelihood,
}

/// Image registration methods
///
/// Registration aligns images from different modalities into a common
/// coordinate system, accounting for patient motion, probe positioning,
/// and geometric distortions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RegistrationMethod {
    /// Rigid body transformation (translation + rotation)
    ///
    /// Assumes no deformation, suitable for bone or rigid anatomy.
    RigidBody,
    /// Affine transformation
    ///
    /// Includes scaling and shearing in addition to rigid transformation.
    Affine,
    /// Non-rigid deformation
    ///
    /// Handles tissue deformation and complex geometric changes.
    NonRigid,
    /// Automatic registration using image features
    ///
    /// Detects and matches features automatically without manual initialization.
    Automatic,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_config_default() {
        let config = FusionConfig::default();
        assert_eq!(config.modality_weights.len(), 3);
        assert!(config.uncertainty_quantification);
        assert_eq!(config.fusion_method, FusionMethod::WeightedAverage);
        assert_eq!(config.registration_method, RegistrationMethod::RigidBody);
        assert_eq!(config.output_resolution, [1e-4, 1e-4, 1e-4]);
    }

    #[test]
    fn test_fusion_config_modality_weights() {
        let config = FusionConfig::default();
        assert!(config.modality_weights.contains_key("ultrasound"));
        assert!(config.modality_weights.contains_key("photoacoustic"));
        assert!(config.modality_weights.contains_key("elastography"));

        let total_weight: f64 = config.modality_weights.values().sum();
        assert!((total_weight - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fusion_method_variants() {
        let methods = [
            FusionMethod::WeightedAverage,
            FusionMethod::FeatureBased,
            FusionMethod::Probabilistic,
            FusionMethod::DeepFusion,
            FusionMethod::MaximumLikelihood,
        ];

        for method in &methods {
            assert!(!format!("{:?}", method).is_empty());
        }
    }

    #[test]
    fn test_registration_method_variants() {
        let methods = [
            RegistrationMethod::RigidBody,
            RegistrationMethod::Affine,
            RegistrationMethod::NonRigid,
            RegistrationMethod::Automatic,
        ];

        for method in &methods {
            assert!(!format!("{:?}", method).is_empty());
        }
    }
}
