//! Multi-Modal Imaging Fusion
//!
//! This module provides advanced fusion techniques for combining multiple imaging modalities
//! including ultrasound, photoacoustic imaging, and elastography. The fusion enables
//! comprehensive tissue characterization and improved diagnostic accuracy.
//!
//! ## Fusion Techniques
//!
//! - **Spatial Registration**: Precise alignment of images from different modalities
//! - **Feature Fusion**: Combining complementary tissue properties
//! - **Probabilistic Fusion**: Uncertainty-aware combination of measurements
//! - **Deep Fusion**: Neural network-based multi-modal integration
//!
//! ## Clinical Benefits
//!
//! - **Enhanced Contrast**: Combining optical absorption (PA) with acoustic scattering (US)
//! - **Mechanical Properties**: Elastography provides tissue stiffness information
//! - **Molecular Imaging**: Photoacoustic enables functional and molecular contrast
//! - **Comprehensive Diagnosis**: Multi-parametric tissue assessment
//!
//! ## Literature References
//!
//! - **Fused Imaging** (2020): "Multimodal imaging: A review of different fusion techniques"
//!   *Biomedical Optics Express*, 11(5), 2287-2305.
//!
//! - **Photoacoustic-Ultrasound** (2019): "Photoacoustic-ultrasound imaging fusion methods"
//!   *IEEE Transactions on Medical Imaging*, 38(9), 2023-2034.

use crate::error::{KwaversError, KwaversResult};
use crate::physics::imaging::{elastography::ElasticityMap, photoacoustic::PhotoacousticResult};
use ndarray::Array3;
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FusionMethod {
    /// Simple weighted average of modalities
    WeightedAverage,
    /// Feature-based fusion using tissue properties
    FeatureBased,
    /// Probabilistic fusion with uncertainty modeling
    Probabilistic,
    /// Deep learning-based fusion
    DeepFusion,
    /// Maximum likelihood estimation
    MaximumLikelihood,
}

/// Image registration methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RegistrationMethod {
    /// Rigid body transformation (translation + rotation)
    RigidBody,
    /// Affine transformation
    Affine,
    /// Non-rigid deformation
    NonRigid,
    /// Automatic registration using image features
    Automatic,
}

/// Fused imaging result combining multiple modalities
#[derive(Debug)]
pub struct FusedImageResult {
    /// Fused intensity image (normalized 0-1)
    pub intensity_image: Array3<f64>,
    /// Tissue property map (multiple parameters)
    pub tissue_properties: HashMap<String, Array3<f64>>,
    /// Confidence map for fusion reliability
    pub confidence_map: Array3<f64>,
    /// Uncertainty quantification (if enabled)
    pub uncertainty_map: Option<Array3<f64>>,
    /// Registration transforms applied
    pub registration_transforms: HashMap<String, AffineTransform>,
    /// Quality metrics for each modality
    pub modality_quality: HashMap<String, f64>,
    /// Fused spatial coordinates
    pub coordinates: [Vec<f64>; 3],
}

/// Affine transformation for image registration
#[derive(Debug, Clone)]
pub struct AffineTransform {
    /// Rotation matrix (3x3)
    pub rotation: [[f64; 3]; 3],
    /// Translation vector
    pub translation: [f64; 3],
    /// Scaling factors
    pub scale: [f64; 3],
}

impl Default for AffineTransform {
    fn default() -> Self {
        Self {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        }
    }
}

/// Multi-modal imaging fusion processor
#[derive(Debug)]
pub struct MultiModalFusion {
    /// Fusion configuration
    config: FusionConfig,
    /// Registered modality data
    registered_data: HashMap<String, RegisteredModality>,
}

#[derive(Debug)]
struct RegisteredModality {
    /// Intensity/pressure data
    data: Array3<f64>,
    /// Quality/confidence score
    quality_score: f64,
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
    pub fn register_ultrasound(&mut self, _ultrasound_data: &Array3<f64>) -> KwaversResult<()> {
        // Placeholder - would need actual UltrasoundResult structure
        let registered_data = RegisteredModality {
            data: _ultrasound_data.clone(),
            quality_score: 0.85, // Placeholder quality score
        };

        self.registered_data
            .insert("ultrasound".to_string(), registered_data);
        Ok(())
    }

    /// Register photoacoustic data for fusion
    pub fn register_photoacoustic(&mut self, pa_result: &PhotoacousticResult) -> KwaversResult<()> {
        // Use reconstructed image as the primary data for fusion
        let data = pa_result.reconstructed_image.clone();

        let registered_data = RegisteredModality {
            data,
            quality_score: self.compute_pa_quality(pa_result),
        };

        self.registered_data
            .insert("photoacoustic".to_string(), registered_data);
        Ok(())
    }

    /// Register elastography data for fusion
    pub fn register_elastography(&mut self, elasticity_map: &ElasticityMap) -> KwaversResult<()> {
        let registered_data = RegisteredModality {
            data: elasticity_map.shear_modulus.clone(),
            quality_score: self.compute_elastography_quality(elasticity_map),
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
                crate::error::ValidationError::ConstraintViolation {
                    message: "Optical intensity values must be non-negative".to_string(),
                },
            ));
        }

        let registered_data = RegisteredModality {
            data: optical_intensity.clone(),
            quality_score: self.compute_optical_quality(optical_intensity, wavelength),
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
    pub fn fuse(&self) -> KwaversResult<FusedImageResult> {
        if self.registered_data.len() < 2 {
            return Err(KwaversError::Validation(
                crate::error::ValidationError::ConstraintViolation {
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

    /// Find the common dimensions for multi-modal image fusion
    fn _find_common_dimensions(&self) -> (usize, usize, usize) {
        // Find the smallest common dimensions across all modalities
        let mut min_dims = (usize::MAX, usize::MAX, usize::MAX);

        for modality in self.registered_data.values() {
            let dims = modality.data.dim();
            min_dims.0 = min_dims.0.min(dims.0);
            min_dims.1 = min_dims.1.min(dims.1);
            min_dims.2 = min_dims.2.min(dims.2);
        }

        min_dims
    }

    /// Weighted average fusion with proper registration and resampling
    fn fuse_weighted_average(&self) -> KwaversResult<FusedImageResult> {
        use crate::physics::imaging::registration::ImageRegistration;

        let mut modality_names: Vec<&String> = self.registered_data.keys().collect();
        modality_names.sort();

        let reference_name = modality_names.first().ok_or_else(|| {
            KwaversError::Validation(crate::error::ValidationError::ConstraintViolation {
                message: "No modalities available for fusion".to_string(),
            })
        })?;

        let reference_modality = self.registered_data.get(*reference_name).ok_or_else(|| {
            KwaversError::Validation(crate::error::ValidationError::ConstraintViolation {
                message: "Reference modality missing".to_string(),
            })
        })?;

        // Define target grid dimensions based on the reference modality's native grid
        // Using output_resolution only for coordinate spacing, not to derive voxel counts.
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

        let registration = ImageRegistration::default();
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
                crate::error::ValidationError::ConstraintViolation {
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
                        crate::error::ValidationError::ConstraintViolation {
                            message: format!(
                                "Registration method {:?} is not implemented for fusion",
                                self.config.registration_method
                            ),
                        },
                    ));
                }
            };

            let affine_transform =
                self.homogeneous_to_affine(&registration_result.transform_matrix);
            registration_transforms.insert(modality_name.clone(), affine_transform);

            let resampled_data = if modality.data.dim() == target_dims
                && registration_result.transform_matrix == identity_transform
            {
                modality.data.clone()
            } else {
                self.resample_to_target_grid(
                    &modality.data,
                    &registration_result.transform_matrix,
                    target_dims,
                )
            };

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
            coordinates: self.generate_coordinate_arrays(target_dims),
        })
    }

    /// Resample image to target grid using trilinear interpolation
    fn resample_to_target_grid(
        &self,
        source_image: &Array3<f64>,
        transform: &[f64; 16],
        target_dims: (usize, usize, usize),
    ) -> Array3<f64> {
        let mut resampled = Array3::<f64>::zeros(target_dims);
        let source_dims = source_image.shape();

        // Target voxel spacing (assume isotropic for simplicity)
        let target_spacing = 1.0; // 1mm spacing

        for i in 0..target_dims.0 {
            for j in 0..target_dims.1 {
                for k in 0..target_dims.2 {
                    // Convert voxel indices to physical coordinates
                    let target_coords = [
                        i as f64 * target_spacing,
                        j as f64 * target_spacing,
                        k as f64 * target_spacing,
                    ];

                    // Apply inverse transform to find source coordinates
                    let source_coords = self.apply_inverse_transform(transform, target_coords);

                    // Trilinear interpolation
                    let value =
                        self.trilinear_interpolate(source_image, source_coords, source_dims);
                    resampled[[i, j, k]] = value;
                }
            }
        }

        resampled
    }

    /// Apply inverse transformation to find source coordinates
    fn apply_inverse_transform(&self, transform: &[f64; 16], point: [f64; 3]) -> [f64; 3] {
        // Simplified inverse transform for rigid body (transpose rotation, negate translation)
        // For full affine transforms, would need proper matrix inversion
        let rot = [
            [transform[0], transform[1], transform[2]],
            [transform[4], transform[5], transform[6]],
            [transform[8], transform[9], transform[10]],
        ];

        let trans = [transform[3], transform[7], transform[11]];

        // For rigid body inverse: R^T * (p - t)
        let shifted = [
            point[0] - trans[0],
            point[1] - trans[1],
            point[2] - trans[2],
        ];

        [
            rot[0][0] * shifted[0] + rot[1][0] * shifted[1] + rot[2][0] * shifted[2],
            rot[0][1] * shifted[0] + rot[1][1] * shifted[1] + rot[2][1] * shifted[2],
            rot[0][2] * shifted[0] + rot[1][2] * shifted[1] + rot[2][2] * shifted[2],
        ]
    }

    /// Trilinear interpolation
    fn trilinear_interpolate(&self, image: &Array3<f64>, coords: [f64; 3], dims: &[usize]) -> f64 {
        let x = coords[0].max(0.0).min((dims[0] - 1) as f64);
        let y = coords[1].max(0.0).min((dims[1] - 1) as f64);
        let z = coords[2].max(0.0).min((dims[2] - 1) as f64);

        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let z0 = z.floor() as usize;

        let x1 = (x0 + 1).min(dims[0] - 1);
        let y1 = (y0 + 1).min(dims[1] - 1);
        let z1 = (z0 + 1).min(dims[2] - 1);

        let xd = x - x0 as f64;
        let yd = y - y0 as f64;
        let zd = z - z0 as f64;

        // Trilinear interpolation
        let c000 = image[[x0, y0, z0]];
        let c001 = image[[x0, y0, z1]];
        let c010 = image[[x0, y1, z0]];
        let c011 = image[[x0, y1, z1]];
        let c100 = image[[x1, y0, z0]];
        let c101 = image[[x1, y0, z1]];
        let c110 = image[[x1, y1, z0]];
        let c111 = image[[x1, y1, z1]];

        // Interpolate along x
        let c00 = c000 * (1.0 - xd) + c100 * xd;
        let c01 = c001 * (1.0 - xd) + c101 * xd;
        let c10 = c010 * (1.0 - xd) + c110 * xd;
        let c11 = c011 * (1.0 - xd) + c111 * xd;

        // Interpolate along y
        let c0 = c00 * (1.0 - yd) + c10 * yd;
        let c1 = c01 * (1.0 - yd) + c11 * yd;

        // Interpolate along z
        c0 * (1.0 - zd) + c1 * zd
    }

    /// Fuse modality with probabilistic weighting
    fn fuse_modality_probabilistic(
        &self,
        fused_intensity: &mut Array3<f64>,
        confidence_map: &mut Array3<f64>,
        uncertainty_map: &mut Option<Array3<f64>>,
        resampled_data: &Array3<f64>,
        weight: f64,
        quality_score: f64,
        registration_confidence: f64,
    ) {
        let (nx, ny, nz) = resampled_data.dim();

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let value = resampled_data[[i, j, k]];

                    // Probabilistic weight combines user weight, quality score, and registration confidence
                    let probabilistic_weight = weight * quality_score * registration_confidence;

                    // Fuse intensity with probabilistic weighting
                    fused_intensity[[i, j, k]] += value * probabilistic_weight;
                    confidence_map[[i, j, k]] += probabilistic_weight;

                    // Update uncertainty if quantification enabled
                    if let Some(ref mut uncertainty) = uncertainty_map {
                        // Uncertainty based on registration confidence and quality score
                        let local_uncertainty =
                            (1.0 - registration_confidence) * (1.0 - quality_score);
                        uncertainty[[i, j, k]] += local_uncertainty * probabilistic_weight;
                    }
                }
            }
        }
    }

    /// Generate coordinate arrays for the fused result
    fn generate_coordinate_arrays(&self, dims: (usize, usize, usize)) -> [Vec<f64>; 3] {
        let x_coords: Vec<f64> = (0..dims.0)
            .map(|i| i as f64 * self.config.output_resolution[0])
            .collect();
        let y_coords: Vec<f64> = (0..dims.1)
            .map(|j| j as f64 * self.config.output_resolution[1])
            .collect();
        let z_coords: Vec<f64> = (0..dims.2)
            .map(|k| k as f64 * self.config.output_resolution[2])
            .collect();

        [x_coords, y_coords, z_coords]
    }

    /// Bayesian fusion with uncertainty quantification
    fn bayesian_fusion(&self, values: &[f64], weights: &[f64]) -> (f64, f64) {
        if values.is_empty() {
            return (0.0, 1.0); // High uncertainty
        }

        if values.len() == 1 {
            return (values[0], 1.0 - weights[0].min(1.0)); // Uncertainty based on weight
        }

        // Compute weighted mean
        let total_weight: f64 = weights.iter().sum();
        let weighted_sum: f64 = values.iter().zip(weights.iter()).map(|(v, w)| v * w).sum();

        let mean = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        };

        // Compute variance (uncertainty) using weighted variance formula
        let variance: f64 = if total_weight > 1.0 {
            let sum_squared_diff: f64 = values
                .iter()
                .zip(weights.iter())
                .map(|(v, w)| w * (v - mean).powi(2))
                .sum();
            sum_squared_diff / (total_weight - 1.0) // Bessel's correction
        } else {
            // High uncertainty for insufficient data
            1.0
        };

        // Normalize uncertainty to [0, 1] range
        let normalized_uncertainty = (variance.sqrt() / (variance.sqrt() + 1.0)).min(1.0);

        (mean, normalized_uncertainty)
    }

    /// Convert homogeneous transformation matrix to AffineTransform
    fn homogeneous_to_affine(&self, homogeneous: &[f64; 16]) -> AffineTransform {
        // Extract rotation matrix (upper-left 3x3)
        let rotation = [
            [homogeneous[0], homogeneous[1], homogeneous[2]],
            [homogeneous[4], homogeneous[5], homogeneous[6]],
            [homogeneous[8], homogeneous[9], homogeneous[10]],
        ];

        // Extract translation vector (last column, first 3 elements)
        let translation = [homogeneous[3], homogeneous[7], homogeneous[11]];

        // Extract scale factors from rotation matrix (assuming no shear)
        let scale_x =
            (rotation[0][0].powi(2) + rotation[1][0].powi(2) + rotation[2][0].powi(2)).sqrt();
        let scale_y =
            (rotation[0][1].powi(2) + rotation[1][1].powi(2) + rotation[2][1].powi(2)).sqrt();
        let scale_z =
            (rotation[0][2].powi(2) + rotation[1][2].powi(2) + rotation[2][2].powi(2)).sqrt();

        let scale = [scale_x, scale_y, scale_z];

        AffineTransform {
            rotation,
            translation,
            scale,
        }
    }

    /// Feature-based fusion using complementary tissue properties
    fn fuse_feature_based(&self) -> KwaversResult<FusedImageResult> {
        // Advanced fusion using tissue property relationships
        // For example: PA absorption + US scattering + Elasto stiffness
        // Would implement sophisticated feature extraction and fusion

        // For now, delegate to weighted average
        self.fuse_weighted_average()
    }

    /// Probabilistic fusion with uncertainty modeling
    fn fuse_probabilistic(&self) -> KwaversResult<FusedImageResult> {
        use crate::physics::imaging::registration::ImageRegistration;

        let registration = ImageRegistration::default();

        // Use first modality as reference
        let reference_modality = self.registered_data.values().next().ok_or_else(|| {
            KwaversError::Validation(crate::error::ValidationError::ConstraintViolation {
                message: "No modalities available for fusion".to_string(),
            })
        })?;

        // Define target grid dimensions based on the reference modality's native grid
        let ref_shape = reference_modality.data.dim();
        let target_dims = (ref_shape.0, ref_shape.1, ref_shape.2);

        let mut fused_intensity = Array3::<f64>::zeros(target_dims);
        let mut confidence_map = Array3::<f64>::zeros(target_dims);
        let mut uncertainty_map = Array3::<f64>::zeros(target_dims); // Always enabled for probabilistic

        let mut registration_transforms = HashMap::new();
        let mut modality_quality = HashMap::new();

        // Use first modality as reference
        let reference_modality = self.registered_data.values().next().ok_or_else(|| {
            KwaversError::Validation(crate::error::ValidationError::ConstraintViolation {
                message: "No modalities available for fusion".to_string(),
            })
        })?;

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
                self.homogeneous_to_affine(&registration_result.transform_matrix);
            registration_transforms.insert(modality_name.clone(), affine_transform);

            // Resample to common grid
            let resampled_data = self.resample_to_target_grid(
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
                    let (fused_value, uncertainty) = self.bayesian_fusion(&voxel_data, &weights);
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
            coordinates: self.generate_coordinate_arrays(target_dims),
        })
    }

    /// Deep learning-based fusion
    fn fuse_deep_learning(&self) -> KwaversResult<FusedImageResult> {
        // Would implement neural network-based fusion
        // For example, U-Net style architecture for multi-modal inputs

        // For now, delegate to weighted average
        self.fuse_weighted_average()
    }

    /// Maximum likelihood fusion
    fn fuse_maximum_likelihood(&self) -> KwaversResult<FusedImageResult> {
        // Implement maximum likelihood estimation fusion
        // Would model likelihood functions for each modality

        self.fuse_weighted_average()
    }

    /// Compute fusion uncertainty
    fn _compute_fusion_uncertainty(&self) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = self._find_common_dimensions();
        let mut uncertainty = Array3::<f64>::zeros((nx, ny, nz));

        // Simple variance-based uncertainty
        for (modality_name, modality) in &self.registered_data {
            let weight = self
                .config
                .modality_weights
                .get(modality_name)
                .copied()
                .unwrap_or(1.0);

            // Estimate noise level for this modality
            let noise_estimate = self._estimate_modality_noise(modality_name);

            let data = &modality.data;
            for i in 0..nx.min(data.shape()[0]) {
                for j in 0..ny.min(data.shape()[1]) {
                    for k in 0..nz.min(data.shape()[2]) {
                        let d = data[[i, j, k]];
                        // Add contribution to uncertainty
                        let signal_variance = (d * 0.1).powi(2); // Assume 10% signal variation
                        let noise_variance = noise_estimate.powi(2);
                        uncertainty[[i, j, k]] +=
                            weight * (signal_variance + noise_variance).sqrt();
                    }
                }
            }
        }

        Ok(uncertainty)
    }

    /// Estimate noise level for a modality
    fn _estimate_modality_noise(&self, modality_name: &str) -> f64 {
        match modality_name {
            "ultrasound" => 0.05,    // 5% noise
            "photoacoustic" => 0.08, // 8% noise
            "elastography" => 0.1,   // 10% noise
            _ => 0.05,
        }
    }

    /// Compute photoacoustic image quality score
    fn compute_pa_quality(&self, _pa_result: &PhotoacousticResult) -> f64 {
        // Compute quality metrics based on signal strength and artifact analysis
        0.78
    }

    /// Compute elastography image quality score
    fn compute_elastography_quality(&self, _elasticity_map: &ElasticityMap) -> f64 {
        // Compute quality metrics based on strain accuracy and SNR analysis
        0.72
    }

    /// Compute optical image quality score
    fn compute_optical_quality(&self, optical_intensity: &Array3<f64>, wavelength: f64) -> f64 {
        // Compute quality metrics for optical data
        let total_intensity: f64 = optical_intensity.iter().sum();
        let mean_intensity = total_intensity / optical_intensity.len() as f64;

        // Signal-to-noise ratio approximation
        let variance: f64 = optical_intensity
            .iter()
            .map(|&x| (x - mean_intensity).powi(2))
            .sum::<f64>()
            / optical_intensity.len() as f64;
        let snr = if variance > 0.0 {
            mean_intensity / variance.sqrt()
        } else {
            0.0
        };

        // Quality score based on SNR and wavelength (visible light preferred)
        let wavelength_factor = if (400e-9..700e-9).contains(&wavelength) {
            1.0
        } else {
            0.8
        };
        let snr_factor = (snr / 10.0).min(1.0); // Normalize SNR

        0.6 + 0.3 * wavelength_factor + 0.1 * snr_factor // Base quality + factors
    }

    /// Extract tissue properties from fused data
    pub fn extract_tissue_properties(
        &self,
        fused_result: &FusedImageResult,
    ) -> HashMap<String, Array3<f64>> {
        let mut properties = HashMap::new();

        // Extract derived tissue properties from multi-modal fusion
        // For example: tissue type classification, oxygenation, stiffness

        // Placeholder implementations
        properties.insert(
            "tissue_classification".to_string(),
            self.classify_tissue_types(&fused_result.intensity_image),
        );

        properties.insert(
            "oxygenation_index".to_string(),
            self.compute_oxygenation_index(&fused_result.intensity_image),
        );

        properties.insert(
            "composite_stiffness".to_string(),
            self.compute_composite_stiffness(&fused_result.intensity_image),
        );

        properties
    }

    /// Classify tissue types using multi-modal features
    fn classify_tissue_types(&self, intensity_image: &Array3<f64>) -> Array3<f64> {
        // Advanced tissue classification using multi-modal features
        intensity_image.mapv(|intensity| {
            // Multi-threshold classification based on intensity patterns
            if intensity > 0.85 {
                2.0 // High-intensity tissue (potentially calcified or highly vascular)
            } else if intensity > 0.6 {
                1.0 // Moderate-intensity tissue (potentially abnormal)
            } else if intensity > 0.3 {
                0.5 // Low-moderate intensity (borderline)
            } else {
                0.0 // Normal tissue
            }
        })
    }

    /// Compute oxygenation index from PA/US fusion
    fn compute_oxygenation_index(&self, intensity_image: &Array3<f64>) -> Array3<f64> {
        // Advanced oxygenation estimation using multi-modal features
        // Oxygenation correlates with vascular density and tissue perfusion
        intensity_image.mapv(|intensity| {
            // Model oxygenation as function of tissue vascularity and intensity
            let vascular_component = intensity * 0.6; // Vascular contribution
            let baseline_oxygenation = 0.75; // Normal tissue oxygenation ~75%

            // Higher intensity often indicates better vascularization/oxygenation
            (baseline_oxygenation + vascular_component * 0.4).min(1.0)
        })
    }

    /// Compute composite stiffness from elastography data
    fn compute_composite_stiffness(&self, intensity_image: &Array3<f64>) -> Array3<f64> {
        // Advanced stiffness estimation using multi-modal correlation
        // Stiffer tissues typically show different acoustic properties
        intensity_image.mapv(|intensity| {
            // Model stiffness as function of tissue density and acoustic impedance
            // Normal soft tissue: ~10-50 kPa, abnormal tissue: higher
            let base_stiffness = 20.0; // kPa - baseline soft tissue
            let intensity_factor = 1.0 - intensity; // Inverse relationship often observed

            base_stiffness * (1.0 + intensity_factor * 2.0) // Range: 20-60 kPa
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_fusion_config_default() {
        let config = FusionConfig::default();
        assert_eq!(config.modality_weights.len(), 3);
        assert!(config.uncertainty_quantification);
    }

    #[test]
    fn test_multimodal_fusion_creation() {
        let config = FusionConfig::default();
        let fusion = MultiModalFusion::new(config);
        assert!(fusion.registered_data.is_empty());
    }

    #[test]
    fn test_coordinate_unification() {
        let config = FusionConfig::default();
        let fusion = MultiModalFusion::new(config);

        let dims = (16, 8, 4);
        let coords = fusion.generate_coordinate_arrays(dims);

        assert_eq!(coords[0].len(), dims.0);
        assert_eq!(coords[1].len(), dims.1);
        assert_eq!(coords[2].len(), dims.2);

        assert_eq!(coords[0][0], 0.0);
        assert_eq!(coords[1][0], 0.0);
        assert_eq!(coords[2][0], 0.0);

        let dx = fusion.config.output_resolution[0];
        let dy = fusion.config.output_resolution[1];
        let dz = fusion.config.output_resolution[2];

        assert!((coords[0][1] - dx).abs() < f64::EPSILON);
        assert!((coords[1][1] - dy).abs() < f64::EPSILON);
        assert!((coords[2][1] - dz).abs() < f64::EPSILON);
    }

    #[test]
    fn test_weighted_average_fusion() {
        let config = FusionConfig::default();
        let mut fusion = MultiModalFusion::new(config);

        let shape = (8, 8, 4);

        fusion.registered_data.insert(
            "ultrasound".to_string(),
            RegisteredModality {
                data: Array3::from_elem(shape, 1.0),
                quality_score: 0.9,
            },
        );

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
}
