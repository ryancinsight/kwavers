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
use ndarray::{Array3, Zip};
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
            FusionMethod::MaximumIntensity => self.fuse_maximum_likelihood(), // TODO: Implement MIP
            FusionMethod::MinimumIntensity => self.fuse_maximum_likelihood(), // TODO: Implement MinIP
            FusionMethod::PCA => self.fuse_maximum_likelihood(), // TODO: Implement PCA fusion
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
        let registration = ImageRegistration::default();

        // 1. Setup Reference and Grid
        // Sort keys to ensure deterministic reference selection
        let mut modality_names: Vec<&String> = self.registered_data.keys().collect();
        modality_names.sort();

        let reference_name = modality_names.first().ok_or_else(|| {
            KwaversError::Validation(crate::core::error::ValidationError::ConstraintViolation {
                message: "No modalities available for fusion".to_string(),
            })
        })?;

        let reference_modality = self
            .registered_data
            .get(*reference_name)
            .ok_or_else(|| KwaversError::InvalidInput("Reference modality missing".to_string()))?;

        let ref_shape = reference_modality.data.dim();
        let target_dims = (ref_shape.0, ref_shape.1, ref_shape.2);

        // 2. Register & Resample All Modalities
        struct Channel {
            name: String,
            data: Array3<f64>,
            quality: f64,
            weight: f64,
        }
        let mut channels = Vec::new();
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];

        let mut registration_transforms = HashMap::new();
        let mut modality_quality = HashMap::new();

        for name in modality_names {
            let modality = self.registered_data.get(name).unwrap();
            let weight = self
                .config
                .modality_weights
                .get(name)
                .copied()
                .unwrap_or(1.0);
            modality_quality.insert(name.clone(), modality.quality_score);

            let reg_result = registration.intensity_registration_mutual_info(
                &reference_modality.data,
                &modality.data,
                &identity,
            )?;

            registration_transforms.insert(
                name.clone(),
                AffineTransform::from_homogeneous(&reg_result.transform_matrix),
            );

            // Optimization: Skip resampling if transform is identity and dimensions match
            let resampled =
                if modality.data.dim() == target_dims && reg_result.transform_matrix == identity {
                    modality.data.clone()
                } else {
                    registration::resample_to_target_grid(
                        &modality.data,
                        &reg_result.transform_matrix,
                        target_dims,
                    )
                };

            channels.push(Channel {
                name: name.clone(),
                data: resampled,
                quality: modality.quality_score,
                weight,
            });
        }

        // 3. Prepare Feature Channels for Classification
        // Pre-compute normalization parameters to avoid full array copies
        struct NormParams {
            min: f64,
            scale: f64,
        }
        let mut norm_params = Vec::new();

        for ch in &channels {
            let (min, max) = compute_robust_bounds(&ch.data);
            let scale = if max > min { 1.0 / (max - min) } else { 0.0 };
            norm_params.push(NormParams { min, scale });
        }

        // 4. Fusion Loop
        let mut fused_intensity = Array3::<f64>::zeros(target_dims);
        let mut confidence_map = Array3::<f64>::zeros(target_dims);
        let mut tissue_class_map = Array3::<f64>::zeros(target_dims);
        let mut classification_confidence_map = Array3::<f64>::zeros(target_dims);

        // Initialize uncertainty map if enabled
        let mut uncertainty_map = if self.config.uncertainty_quantification {
            Some(Array3::<f64>::zeros(target_dims))
        } else {
            None
        };

        for i in 0..target_dims.0 {
            for j in 0..target_dims.1 {
                for k in 0..target_dims.2 {
                    // Extract normalized features for classification on-the-fly
                    let mut us_val = 0.5; // Default mid
                    let mut pa_val = 0.0; // Default low
                    let mut stiff_val = 0.5; // Default mid

                    for (idx, ch) in channels.iter().enumerate() {
                        let val = ch.data[[i, j, k]];
                        let norm_val = ((val - norm_params[idx].min) * norm_params[idx].scale)
                            .max(0.0)
                            .min(1.0);

                        if ch.name.contains("ultrasound") {
                            us_val = norm_val;
                        } else if ch.name.contains("photoacoustic") {
                            pa_val = norm_val;
                        } else if ch.name.contains("elastography") {
                            stiff_val = norm_val;
                        }
                    }

                    let (tissue_type, class_conf) =
                        classify_voxel_features(us_val, pa_val, stiff_val);
                    tissue_class_map[[i, j, k]] = tissue_type as f64;
                    classification_confidence_map[[i, j, k]] = class_conf;

                    // Compute Adaptive Weights
                    let mut sum_weighted_val = 0.0;
                    let mut sum_weights = 0.0;
                    let mut sum_uncertainty = 0.0;

                    for ch in &channels {
                        let val = ch.data[[i, j, k]];

                        // Base weight from config and quality
                        let mut w = ch.weight * ch.quality;

                        // Adaptive adjustment based on tissue type
                        // 0: Fluid, 1: Soft, 2: Vessel, 3: Hard
                        match tissue_type {
                            0 => {
                                // Fluid: Trust US, distrust PA (noise)
                                if ch.name.contains("ultrasound") {
                                    w *= 1.5;
                                }
                                if ch.name.contains("photoacoustic") {
                                    w *= 0.5;
                                }
                            }
                            2 => {
                                // Vessel: Trust PA highly
                                if ch.name.contains("photoacoustic") {
                                    w *= 2.0;
                                }
                            }
                            3 => {
                                // Hard/Calcified: Trust Elastography and US
                                if ch.name.contains("elastography") {
                                    w *= 1.5;
                                }
                                if ch.name.contains("ultrasound") {
                                    w *= 1.2;
                                }
                            }
                            _ => {} // Soft tissue: default weights
                        }

                        sum_weighted_val += val * w;
                        sum_weights += w;

                        if let Some(_) = uncertainty_map {
                            sum_uncertainty += (1.0 - ch.quality) * w;
                        }
                    }

                    if sum_weights > 0.0 {
                        fused_intensity[[i, j, k]] = sum_weighted_val / sum_weights;
                        confidence_map[[i, j, k]] = sum_weights; // Proxy for confidence

                        if let Some(ref mut u_map) = uncertainty_map {
                            u_map[[i, j, k]] = sum_uncertainty / sum_weights;
                        }
                    }
                }
            }
        }

        let mut tissue_properties = HashMap::new();
        tissue_properties.insert("tissue_classification".to_string(), tissue_class_map);
        tissue_properties.insert(
            "classification_confidence".to_string(),
            classification_confidence_map,
        );

        Ok(FusedImageResult {
            intensity_image: fused_intensity,
            tissue_properties,
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
        let image_reg = ImageRegistration::default();

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
        let mut uncertainty_map = Array3::<f64>::zeros(target_dims);

        let mut registration_transforms = HashMap::new();
        let mut modality_quality = HashMap::new();

        // 1. Prepare and register all data
        // We store: (resampled_data, current_variance)
        struct ModalityContext {
            data: Array3<f64>,
            variance: f64,
        }
        let mut contexts = Vec::new();

        let identity_transform = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];

        for modality_name in &modality_names {
            let modality = self.registered_data.get(modality_name.as_str()).unwrap();
            modality_quality.insert(modality_name.to_string(), modality.quality_score);

            // Register
            let registration_result = image_reg.intensity_registration_mutual_info(
                &reference_modality.data,
                &modality.data,
                &identity_transform,
            )?;

            let affine_transform =
                AffineTransform::from_homogeneous(&registration_result.transform_matrix);
            registration_transforms.insert(modality_name.to_string(), affine_transform);

            // Resample
            let resampled_data = registration::resample_to_target_grid(
                &modality.data,
                &registration_result.transform_matrix,
                target_dims,
            );

            // Initialize variance based on quality score
            // Higher quality -> lower variance
            // Avoid division by zero with small epsilon
            let initial_variance = 1.0 / (modality.quality_score + 1e-6);

            contexts.push(ModalityContext {
                data: resampled_data,
                variance: initial_variance,
            });
        }

        // 2. EM Algorithm Loop
        const MAX_ITERATIONS: usize = 10;
        const CONVERGENCE_THRESHOLD: f64 = 1e-6;
        const MIN_VARIANCE: f64 = 1e-9;
        let num_voxels = (target_dims.0 * target_dims.1 * target_dims.2) as f64;

        for _iter in 0..MAX_ITERATIONS {
            let mut max_change = 0.0;

            // E-Step: Estimate fused image using current variances
            // fused = sum(data_i / var_i) / sum(1 / var_i)
            let mut numerator = Array3::<f64>::zeros(target_dims);
            let mut denominator = 0.0;

            for ctx in &contexts {
                let w = 1.0 / ctx.variance;
                // Accumulate weighted data in place to avoid allocation
                numerator.scaled_add(w, &ctx.data);
                denominator += w;
            }

            let new_fused_intensity = numerator.mapv(|x| x / denominator);

            // Check convergence on image
            // We can use a simplified check: L2 norm difference or max difference
            // Here using max difference
            let diff = &new_fused_intensity - &fused_intensity;
            for v in diff.iter() {
                if v.abs() > max_change {
                    max_change = v.abs();
                }
            }

            fused_intensity = new_fused_intensity;

            if max_change < CONVERGENCE_THRESHOLD {
                break;
            }

            // M-Step: Update variances
            // var_i = mean((data_i - fused)^2)
            for ctx in &mut contexts {
                let sum_sq_error: f64 =
                    Zip::from(&ctx.data)
                        .and(&fused_intensity)
                        .fold(0.0, |acc, &val, &mean| {
                            let diff = val - mean;
                            acc + diff * diff
                        });
                ctx.variance = (sum_sq_error / num_voxels).max(MIN_VARIANCE);
            }
        }

        // 3. Finalize Uncertainty and Confidence
        // Cramér-Rao Bound (CRB): 1 / sum(1/var_i)
        // This corresponds to the variance of the weighted mean estimator
        let mut total_fisher_info = 0.0;
        for ctx in &contexts {
            total_fisher_info += 1.0 / ctx.variance;
        }
        let crb = 1.0 / total_fisher_info;

        // Populate maps (uniform since our noise model is homoscedastic per modality)
        uncertainty_map.fill(crb);
        confidence_map.fill(total_fisher_info); // Fisher info as confidence measure

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

/// Compute robust normalization bounds (1st and 99th percentiles)
fn compute_robust_bounds(data: &Array3<f64>) -> (f64, f64) {
    let mut values: Vec<f64> = data.iter().cloned().filter(|v| v.is_finite()).collect();

    if values.is_empty() {
        return (0.0, 0.0);
    }

    values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let len = values.len();
    let lower_idx = (len as f64 * 0.01).floor() as usize;
    let upper_idx = (len as f64 * 0.99).ceil() as usize;

    let min_val = values[lower_idx.min(len - 1)];
    let max_val = values[upper_idx.min(len - 1)];

    if max_val <= min_val {
        (min_val, min_val + 1.0) // Avoid division by zero
    } else {
        (min_val, max_val)
    }
}

/// Classify tissue type based on multi-modal features
///
/// Returns (TissueType, Confidence)
/// Tissue Types:
/// 0: Fluid/Background (Low US, Low Stiffness)
/// 1: Soft Tissue (Mid US, Mid Stiffness)
/// 2: Vessel/Blood (High PA, Low Stiffness)
/// 3: Hard Tissue/Calcification (High US, High Stiffness)
fn classify_voxel_features(us: f64, pa: f64, stiffness: f64) -> (u8, f64) {
    // Heuristic classification logic

    // Check for Hard Tissue (High Stiffness is the strongest indicator)
    if stiffness > 0.7 {
        if us > 0.6 {
            return (3, stiffness * us); // High US + High Stiffness -> Calcification/Bone
        }
        return (3, stiffness * 0.8); // High Stiffness -> Hard Tissue
    }

    // Check for Vessel/Blood (High PA is key)
    if pa > 0.6 {
        if stiffness < 0.4 {
            return (2, pa * (1.0 - stiffness)); // High PA + Low Stiffness -> Vessel
        }
        return (2, pa * 0.7); // High PA -> Likely vessel
    }

    // Check for Fluid (Low signal across board, esp stiffness and US)
    if us < 0.3 && stiffness < 0.3 {
        return (0, (1.0 - us) * (1.0 - stiffness));
    }

    // Default to Soft Tissue
    (1, 0.5)
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

    #[test]
    fn test_maximum_likelihood_fusion() {
        let config = FusionConfig {
            fusion_method: FusionMethod::MaximumLikelihood,
            ..Default::default()
        };
        let mut fusion = MultiModalFusion::new(config);

        let shape = (10, 10, 2);

        // Modality 1: "Clean", high quality. Value = 1.0.
        // Expect low variance.
        fusion.registered_data.insert(
            "modality_clean".to_string(),
            RegisteredModality {
                data: Array3::from_elem(shape, 1.0),
                quality_score: 0.95,
            },
        );

        // Modality 2: "Noisy", low quality. Value = 2.0.
        // Expect high variance.
        fusion.registered_data.insert(
            "modality_noisy".to_string(),
            RegisteredModality {
                data: Array3::from_elem(shape, 2.0),
                quality_score: 0.05,
            },
        );

        let fused = fusion.fuse().unwrap();

        // 1. Verify dimensions
        assert_eq!(fused.intensity_image.dim(), shape);

        // 2. Verify fused value
        // The fused value should be closer to 1.0 than 2.0 because "modality_clean" has higher quality (lower variance).
        // Initial variance ~ 1/quality.
        // var_clean ~ 1/0.95 = 1.05
        // var_noisy ~ 1/0.05 = 20.0
        // weight_clean ~ 0.95, weight_noisy ~ 0.05
        // Expected mean ~ (1.0 * 0.95 + 2.0 * 0.05) / (0.95 + 0.05) = (0.95 + 0.1) / 1.0 = 1.05
        // The actual EM might shift variances, but the trend should hold.
        let val = fused.intensity_image[[0, 0, 0]];
        println!("Fused value: {}", val);
        assert!(val < 1.2, "Fused value {} should be close to 1.0", val);

        // 3. Verify uncertainty
        // Should be populated
        assert!(fused.uncertainty_map.is_some());
        let uncertainty = fused.uncertainty_map.as_ref().unwrap();
        let u_val = uncertainty[[0, 0, 0]];
        assert!(u_val > 0.0);

        // CRB = 1 / (1/var_clean + 1/var_noisy)
        // CRB < var_clean
        // So uncertainty should be small
    }

    #[test]
    fn test_fuse_feature_based_vessel() {
        let mut config = FusionConfig::default();
        config.fusion_method = FusionMethod::FeatureBased;
        config.uncertainty_quantification = false; // Simplify test

        let mut fusion = MultiModalFusion::new(config);
        let shape = (4, 4, 2);

        // Create data with full range [0, 1] to ensure normalization works as expected
        // US: Low signal (0.2)
        let mut us_data = Array3::<f64>::zeros(shape);
        us_data[[0, 0, 0]] = 0.0;
        us_data[[3, 3, 1]] = 1.0;
        us_data[[1, 1, 0]] = 0.2;

        // PA: High signal (0.8) -> Vessel indicator
        let mut pa_data = Array3::<f64>::zeros(shape);
        pa_data[[0, 0, 0]] = 0.0;
        pa_data[[3, 3, 1]] = 1.0;
        pa_data[[1, 1, 0]] = 0.8;

        // Elasto: Low stiffness (0.3) -> Soft/Vessel
        let mut elasto_data = Array3::<f64>::zeros(shape);
        elasto_data[[0, 0, 0]] = 0.0;
        elasto_data[[3, 3, 1]] = 1.0;
        elasto_data[[1, 1, 0]] = 0.3;

        fusion.register_ultrasound(&us_data).unwrap();
        fusion.register_photoacoustic(&pa_data).unwrap();
        fusion
            .register_elastography(
                &crate::domain::imaging::ultrasound::elastography::ElasticityMap {
                    youngs_modulus: Array3::zeros(shape),
                    shear_modulus: elasto_data,
                    shear_wave_speed: Array3::zeros(shape),
                },
            )
            .unwrap();

        let result = fusion.fuse().unwrap();

        // Check classification at [1,1,0]
        // PA > 0.6, Stiffness < 0.4 => Type 2 (Vessel)
        let classification = result
            .tissue_properties
            .get("tissue_classification")
            .unwrap();
        assert_eq!(classification[[1, 1, 0]], 2.0);

        // Check confidence is populated
        assert!(result
            .tissue_properties
            .contains_key("classification_confidence"));
    }
}
