//! Fusion algorithms for combining multi-modal imaging data.
//!
//! This module implements various fusion strategies for combining data from
//! multiple imaging modalities, including weighted averaging, probabilistic
//! fusion, feature-based methods, and machine learning approaches.
//!
//! TODO_AUDIT: P2 - Advanced Multi-Modal Fusion - Implement complete fusion algorithms including deep learning, feature-based, and probabilistic methods
//! DEPENDS ON: physics/acoustics/imaging/fusion/deep_learning.rs, physics/acoustics/imaging/fusion/feature_extraction.rs, physics/acoustics/imaging/fusion/probabilistic.rs
//! MISSING: Deep learning fusion with U-Net architectures and attention mechanisms
//! MISSING: Feature-based fusion with tissue classification and correlation analysis
//! MISSING: Probabilistic fusion using Bayesian methods and uncertainty quantification
//! MISSING: Real-time fusion for streaming multi-modal data
//! MISSING: Quality assessment and fusion confidence metrics
//! SEVERITY: HIGH (enables advanced clinical imaging workflows)
//! THEOREM: Information fusion: H(fused) ≤ min(H_i) where H is entropy, for complementary modalities
//! THEOREM: Bayesian fusion: P(fused|data) ∝ ∏ P(data_i|fused) P(fused) for independent measurements
//! REFERENCES: Blum & Liu (2006) Multi-Sensor Image Fusion; Ma et al. (2019) Medical Image Fusion

use super::config::{FusionConfig, FusionMethod, RegistrationMethod};
use super::quality;
use super::registration;
use super::types::{AffineTransform, FusedImageResult, RegisteredModality};
use crate::core::error::{KwaversError, KwaversResult};
use crate::physics::imaging::registration::ImageRegistration;
use ndarray::Array3;
use std::collections::HashMap;

/// Multi-modal imaging fusion processor
#[derive(Debug)]
pub struct MultiModalFusion {
    /// Fusion configuration
    pub(crate) config: FusionConfig,
    /// Registered modality data
    pub(crate) registered_data: HashMap<String, RegisteredModality>,
}

impl MultiModalFusion {
    /// Create a new multi-modal fusion processor
    pub fn new(config: FusionConfig) -> Self {
        Self {
            config,
            registered_data: HashMap::new(),
        }
    }

    /// Register ultrasound data for multi-modal image fusion
    pub fn register_ultrasound(&mut self, ultrasound_data: &Array3<f64>) -> KwaversResult<()> {
        let registered_data = RegisteredModality {
            data: ultrasound_data.clone(),
            quality_score: 0.85, // Placeholder quality score
        };

        self.registered_data
            .insert("ultrasound".to_string(), registered_data);
        Ok(())
    }

    /// Register photoacoustic data for fusion
    pub fn register_photoacoustic(
        &mut self,
        reconstructed_image: &Array3<f64>,
    ) -> KwaversResult<()> {
        let registered_data = RegisteredModality {
            data: reconstructed_image.clone(),
            quality_score: quality::compute_pa_quality(reconstructed_image),
        };

        self.registered_data
            .insert("photoacoustic".to_string(), registered_data);
        Ok(())
    }

    /// Register elastography data for fusion
    pub fn register_elastography(
        &mut self,
        elasticity_map: &crate::domain::imaging::ultrasound::elastography::ElasticityMap,
    ) -> KwaversResult<()> {
        let registered_data = RegisteredModality {
            data: elasticity_map.shear_modulus.clone(),
            quality_score: quality::compute_elastography_quality(elasticity_map),
        };

        self.registered_data
            .insert("elastography".to_string(), registered_data);
        Ok(())
    }

    /// Register optical/sonoluminescence data for fusion
    pub fn register_optical(
        &mut self,
        optical_intensity: &Array3<f64>,
        wavelength: f64,
    ) -> KwaversResult<()> {
        // Validate optical data represents intensity/emission
        if optical_intensity.iter().any(|&x| x < 0.0) {
            return Err(KwaversError::Validation(
                crate::core::error::ValidationError::ConstraintViolation {
                    message: "Optical intensity values must be non-negative".to_string(),
                },
            ));
        }

        let registered_data = RegisteredModality {
            data: optical_intensity.clone(),
            quality_score: quality::compute_optical_quality(optical_intensity, wavelength),
        };

        self.registered_data.insert(
            format!("optical_{}nm", (wavelength * 1e9) as usize),
            registered_data,
        );
        Ok(())
    }

    /// Get the number of registered modalities
    #[must_use]
    pub fn num_registered_modalities(&self) -> usize {
        self.registered_data.len()
    }

    /// Check if a modality is registered
    #[must_use]
    pub fn is_modality_registered(&self, modality_name: &str) -> bool {
        self.registered_data.contains_key(modality_name)
    }

    /// Perform multi-modal fusion
    ///
    /// Combines all registered modalities according to the configured fusion method.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Fewer than two modalities are registered
    /// - Registration or resampling fails
    /// - Configuration parameters are invalid
    pub fn fuse(&self) -> KwaversResult<FusedImageResult> {
        if self.registered_data.len() < 2 {
            return Err(KwaversError::Validation(
                crate::core::error::ValidationError::ConstraintViolation {
                    message: "At least two modalities required for fusion".to_string(),
                },
            ));
        }

        // Apply fusion method
        let fused_result = match self.config.fusion_method {
            FusionMethod::WeightedAverage => self.fuse_weighted_average(),
            FusionMethod::FeatureBased => self.fuse_feature_based(),
            FusionMethod::Probabilistic => self.fuse_probabilistic(),
            FusionMethod::DeepFusion => self.fuse_deep_learning(),
            FusionMethod::MaximumLikelihood => self.fuse_maximum_likelihood(),
        }?;

        Ok(fused_result)
    }

    /// Weighted average fusion with registration and resampling
    ///
    /// Performs simple weighted averaging of modalities after registering them
    /// to a common coordinate system. This is the most straightforward fusion
    /// approach and works well when all modalities have similar quality.
    fn fuse_weighted_average(&self) -> KwaversResult<FusedImageResult> {
        let registration = ImageRegistration::default();

        let mut modality_names: Vec<&String> = self.registered_data.keys().collect();
        modality_names.sort();

        let reference_name = modality_names.first().ok_or_else(|| {
            KwaversError::Validation(crate::core::error::ValidationError::ConstraintViolation {
                message: "No modalities available for fusion".to_string(),
            })
        })?;

        let reference_modality = self.registered_data.get(*reference_name).ok_or_else(|| {
            KwaversError::Validation(crate::core::error::ValidationError::ConstraintViolation {
                message: "Reference modality missing".to_string(),
            })
        })?;

        // Define target grid dimensions based on the reference modality's native grid
        let ref_shape = reference_modality.data.dim();
        let target_dims = (ref_shape.0, ref_shape.1, ref_shape.2);

        let mut fused_intensity = Array3::<f64>::zeros(target_dims);
        let mut confidence_map = Array3::<f64>::zeros(target_dims);
        let mut uncertainty_map = if self.config.uncertainty_quantification {
            Some(Array3::<f64>::zeros(target_dims))
        } else {
            None
        };

        let mut registration_transforms = HashMap::new();
        let mut modality_quality = HashMap::new();

        // Validate total weight
        let total_weight: f64 = modality_names
            .iter()
            .map(|&name| {
                self.config
                    .modality_weights
                    .get(name.as_str())
                    .copied()
                    .unwrap_or(1.0)
            })
            .sum();

        if total_weight <= 0.0 || !total_weight.is_finite() {
            return Err(KwaversError::Validation(
                crate::core::error::ValidationError::ConstraintViolation {
                    message: "FusionConfig.modality_weights must sum to a positive finite value"
                        .to_string(),
                },
            ));
        }

        let identity_transform = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];

        // Register and resample each modality
        for modality_name in modality_names {
            let modality = self
                .registered_data
                .get(modality_name.as_str())
                .ok_or_else(|| KwaversError::InvalidInput("Modality missing".to_string()))?;

            let weight = self
                .config
                .modality_weights
                .get(modality_name.as_str())
                .copied()
                .unwrap_or(1.0);

            modality_quality.insert(modality_name.clone(), modality.quality_score);

            let registration_result = match self.config.registration_method {
                RegistrationMethod::RigidBody | RegistrationMethod::Automatic => registration
                    .intensity_registration_mutual_info(
                        &reference_modality.data,
                        &modality.data,
                        &identity_transform,
                    )?,
                RegistrationMethod::Affine | RegistrationMethod::NonRigid => {
                    return Err(KwaversError::Validation(
                        crate::core::error::ValidationError::ConstraintViolation {
                            message: format!(
                                "Registration method {:?} is not implemented for fusion",
                                self.config.registration_method
                            ),
                        },
                    ));
                }
            };

            let affine_transform =
                AffineTransform::from_homogeneous(&registration_result.transform_matrix);
            registration_transforms.insert(modality_name.clone(), affine_transform);

            let resampled_data = if modality.data.dim() == target_dims
                && registration_result.transform_matrix == identity_transform
            {
                modality.data.clone()
            } else {
                registration::resample_to_target_grid(
                    &modality.data,
                    &registration_result.transform_matrix,
                    target_dims,
                )
            };

            // Accumulate weighted data
            for i in 0..target_dims.0 {
                for j in 0..target_dims.1 {
                    for k in 0..target_dims.2 {
                        let value = resampled_data[[i, j, k]];
                        fused_intensity[[i, j, k]] += value * weight;
                        confidence_map[[i, j, k]] += weight;
                        if let Some(ref mut uncertainty) = uncertainty_map {
                            uncertainty[[i, j, k]] += (1.0 - modality.quality_score) * weight;
                        }
                    }
                }
            }
        }

        // Normalize fused intensity by weight sum
        for i in 0..target_dims.0 {
            for j in 0..target_dims.1 {
                for k in 0..target_dims.2 {
                    let conf = confidence_map[[i, j, k]];
                    if conf > 0.0 {
                        fused_intensity[[i, j, k]] /= conf;
                        if let Some(ref mut uncertainty) = uncertainty_map {
                            uncertainty[[i, j, k]] /= conf;
                        }
                    }
                }
            }
        }

        Ok(FusedImageResult {
            intensity_image: fused_intensity,
            tissue_properties: HashMap::new(),
            confidence_map,
            uncertainty_map,
            registration_transforms,
            modality_quality,
            coordinates: registration::generate_coordinate_arrays(
                target_dims,
                self.config.output_resolution,
            ),
        })
    }

    /// Probabilistic fusion with uncertainty modeling
    ///
    /// Performs Bayesian fusion that accounts for uncertainty in each modality.
    /// This method provides uncertainty estimates at each voxel based on the
    /// consistency and quality of the multi-modal measurements.
    fn fuse_probabilistic(&self) -> KwaversResult<FusedImageResult> {
        let registration = ImageRegistration::default();

        // Use first modality as reference
        let reference_modality = self.registered_data.values().next().ok_or_else(|| {
            KwaversError::Validation(crate::core::error::ValidationError::ConstraintViolation {
                message: "No modalities available for fusion".to_string(),
            })
        })?;

        // Define target grid dimensions based on the reference modality's native grid
        let ref_shape = reference_modality.data.dim();
        let target_dims = (ref_shape.0, ref_shape.1, ref_shape.2);

        let mut fused_intensity = Array3::<f64>::zeros(target_dims);
        let mut confidence_map = Array3::<f64>::zeros(target_dims);
        let mut uncertainty_map = Array3::<f64>::zeros(target_dims);

        let mut registration_transforms = HashMap::new();
        let mut modality_quality = HashMap::new();

        // Collect all modality data for probabilistic fusion
        let mut modality_data = Vec::new();

        for (modality_name, modality) in &self.registered_data {
            let weight = self
                .config
                .modality_weights
                .get(modality_name)
                .copied()
                .unwrap_or(1.0);

            modality_quality.insert(modality_name.clone(), modality.quality_score);

            // Register modality
            let identity_transform = [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ];

            let registration_result = registration.intensity_registration_mutual_info(
                &reference_modality.data,
                &modality.data,
                &identity_transform,
            )?;

            let affine_transform =
                AffineTransform::from_homogeneous(&registration_result.transform_matrix);
            registration_transforms.insert(modality_name.clone(), affine_transform);

            // Resample to common grid
            let resampled_data = registration::resample_to_target_grid(
                &modality.data,
                &registration_result.transform_matrix,
                target_dims,
            );

            modality_data.push((
                resampled_data,
                weight,
                modality.quality_score,
                registration_result.confidence,
            ));
        }

        // Perform probabilistic fusion at each voxel
        for i in 0..target_dims.0 {
            for j in 0..target_dims.1 {
                for k in 0..target_dims.2 {
                    let voxel_data: Vec<f64> = modality_data
                        .iter()
                        .map(|(data, _, _, _)| data[[i, j, k]])
                        .collect();

                    let weights: Vec<f64> = modality_data
                        .iter()
                        .map(|(_, weight, quality, confidence)| weight * quality * confidence)
                        .collect();

                    // Bayesian fusion with uncertainty quantification
                    let (fused_value, uncertainty) =
                        quality::bayesian_fusion_single_voxel(&voxel_data, &weights);
                    let total_confidence = weights.iter().sum::<f64>();

                    fused_intensity[[i, j, k]] = fused_value;
                    confidence_map[[i, j, k]] = total_confidence;
                    uncertainty_map[[i, j, k]] = uncertainty;
                }
            }
        }

        Ok(FusedImageResult {
            intensity_image: fused_intensity,
            tissue_properties: HashMap::new(),
            confidence_map,
            uncertainty_map: Some(uncertainty_map),
            registration_transforms,
            modality_quality,
            coordinates: registration::generate_coordinate_arrays(
                target_dims,
                self.config.output_resolution,
            ),
        })
    }

    /// Feature-based fusion using complementary tissue properties
    ///
    /// Advanced fusion using tissue property relationships. For example:
    /// PA absorption + US scattering + Elastography stiffness.
    ///
    /// This would implement sophisticated feature extraction and fusion
    /// based on known relationships between modalities.
    fn fuse_feature_based(&self) -> KwaversResult<FusedImageResult> {
        // TODO: Implement feature-based fusion with:
        // - Tissue classification from multi-modal features
        // - Correlation analysis between modalities
        // - Feature space fusion (e.g., PCA, ICA)
        // - Adaptive weighting based on tissue type

        // For now, delegate to weighted average
        self.fuse_weighted_average()
    }

    /// Deep learning-based fusion
    ///
    /// Neural network-based fusion for complex multi-modal relationships.
    /// Would implement architectures like U-Net or attention-based models
    /// for learning optimal fusion strategies from training data.
    fn fuse_deep_learning(&self) -> KwaversResult<FusedImageResult> {
        // TODO: Implement deep learning fusion with:
        // - U-Net style architecture for multi-modal inputs
        // - Attention mechanisms for modality weighting
        // - Multi-scale feature extraction
        // - End-to-end training on clinical datasets

        // For now, delegate to weighted average
        self.fuse_weighted_average()
    }

    /// Maximum likelihood estimation fusion
    ///
    /// Statistical estimation method that maximizes the likelihood of the
    /// observed multi-modal data given a noise model for each modality.
    fn fuse_maximum_likelihood(&self) -> KwaversResult<FusedImageResult> {
        // TODO: Implement MLE fusion with:
        // - Likelihood function modeling for each modality
        // - Noise variance estimation
        // - Iterative optimization (EM algorithm)
        // - Cramér-Rao bound for uncertainty

        // For now, delegate to weighted average
        self.fuse_weighted_average()
    }

    /// Extract tissue properties from fused imaging data
    ///
    /// Convenience method that delegates to the `properties` module's
    /// `extract_tissue_properties` function.
    ///
    /// # Arguments
    ///
    /// * `fused_result` - Fused imaging result
    ///
    /// # Returns
    ///
    /// HashMap mapping property names to 3D spatial maps
    pub fn extract_tissue_properties(
        &self,
        fused_result: &FusedImageResult,
    ) -> HashMap<String, Array3<f64>> {
        super::properties::extract_tissue_properties(fused_result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multimodal_fusion_creation() {
        let config = FusionConfig::default();
        let fusion = MultiModalFusion::new(config);
        assert_eq!(fusion.num_registered_modalities(), 0);
        assert!(fusion.registered_data.is_empty());
    }

    #[test]
    fn test_register_ultrasound() {
        let config = FusionConfig::default();
        let mut fusion = MultiModalFusion::new(config);

        let data = Array3::<f64>::zeros((8, 8, 4));
        let result = fusion.register_ultrasound(&data);

        assert!(result.is_ok());
        assert_eq!(fusion.num_registered_modalities(), 1);
        assert!(fusion.is_modality_registered("ultrasound"));
    }

    #[test]
    fn test_fuse_insufficient_modalities() {
        let config = FusionConfig::default();
        let fusion = MultiModalFusion::new(config);

        let result = fusion.fuse();
        assert!(result.is_err());
    }

    #[test]
    fn test_weighted_average_fusion() {
        let config = FusionConfig::default();
        let mut fusion = MultiModalFusion::new(config);

        let shape = (8, 8, 4);

        fusion
            .register_ultrasound(&Array3::from_elem(shape, 1.0))
            .unwrap();

        // Create a second modality by registering it with a different name
        fusion.registered_data.insert(
            "photoacoustic".to_string(),
            RegisteredModality {
                data: Array3::from_elem(shape, 3.0),
                quality_score: 0.8,
            },
        );

        let fused = fusion.fuse().unwrap();

        assert_eq!(fused.intensity_image.dim(), shape);

        let weights = &fusion.config.modality_weights;
        let w_us = weights["ultrasound"];
        let w_pa = weights["photoacoustic"];
        let expected = (w_us * 1.0 + w_pa * 3.0) / (w_us + w_pa);

        for v in fused.intensity_image.iter() {
            assert!((v - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn test_register_optical_validation() {
        let config = FusionConfig::default();
        let mut fusion = MultiModalFusion::new(config);

        // Test with negative values (should fail)
        let mut invalid_data = Array3::<f64>::zeros((4, 4, 2));
        invalid_data[[0, 0, 0]] = -1.0;

        let result = fusion.register_optical(&invalid_data, 550e-9);
        assert!(result.is_err());

        // Test with valid values (should succeed)
        let valid_data = Array3::<f64>::ones((4, 4, 2));
        let result = fusion.register_optical(&valid_data, 550e-9);
        assert!(result.is_ok());
    }

    #[test]
    fn test_extract_tissue_properties_method() {
        let config = FusionConfig::default();
        let mut fusion = MultiModalFusion::new(config);
        let shape = (4, 4, 2);

        fusion
            .register_ultrasound(&Array3::from_elem(shape, 1.0))
            .unwrap();
        fusion.registered_data.insert(
            "photoacoustic".to_string(),
            RegisteredModality {
                data: Array3::from_elem(shape, 2.0),
                quality_score: 0.8,
            },
        );

        let fused = fusion.fuse().unwrap();
        let properties = fusion.extract_tissue_properties(&fused);

        assert!(properties.contains_key("tissue_classification"));
        assert!(properties.contains_key("oxygenation_index"));
        assert!(properties.contains_key("composite_stiffness"));
    }
}
