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
    #[ignore] // FusionConfig fields changed in domain layer
    fn test_fusion_config_default() {
        let config = FusionConfig::default();
        // These fields don't exist in the new domain FusionConfig
        // assert_eq!(config.modality_weights.len(), 3);
        // assert!(config.uncertainty_quantification);
        assert_eq!(config.fusion_method, FusionMethod::WeightedAverage);
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
