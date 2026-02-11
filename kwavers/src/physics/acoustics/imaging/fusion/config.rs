//! Configuration types for multi-modal imaging fusion
//!
//! Note: FusionConfig and FusionMethod have been moved to domain::imaging::fusion
//! for clean architecture compliance. This module now only contains physics-specific
//! registration configuration.
//!
//! This module defines the configuration structures and enums that control
//! how different imaging modalities are combined, including fusion methods,
//! registration strategies, and quality parameters.

// Re-export from domain layer (SSOT for data models)
pub use crate::domain::imaging::fusion::{FusionConfig, FusionMethod, RegistrationMethod};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_config_default() {
        let config = FusionConfig::default();
        // Verify default fusion method
        assert_eq!(config.fusion_method, FusionMethod::WeightedAverage);
        // Verify default registration method
        assert_eq!(config.registration_method, RegistrationMethod::RigidBody);
        // Verify modality weights map is empty initially
        assert!(config.modality_weights.is_empty());
        // Verify default uncertainty quantification setting
        assert_eq!(config.uncertainty_quantification, false);
        // Verify default quality threshold
        assert_eq!(config.min_quality_threshold, 0.3);
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
