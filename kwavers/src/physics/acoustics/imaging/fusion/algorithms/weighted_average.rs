use super::MultiModalFusion;
use crate::core::error::{KwaversError, KwaversResult};
use crate::physics::acoustics::imaging::fusion::registration;
use crate::physics::acoustics::imaging::fusion::types::{AffineTransform, FusedImageResult};
use crate::physics::imaging::registration::ImageRegistration;
use ndarray::{Array3, CowArray};
use std::collections::HashMap;

/// Weighted average fusion with registration and resampling
///
/// Performs simple weighted averaging of modalities after registering them
/// to a common coordinate system. This is the most straightforward fusion
/// approach and works well when all modalities have similar quality.
///
/// # Mathematical Specification
///
/// Let $I_k(\vec{x})$ be the intensity of modality $k$ at spatial coordinate $\vec{x}$.
/// The fused intensity $F(\vec{x})$ is given by the weighted sum:
///
/// $$ F(\vec{x}) = \frac{\sum_{k=1}^N w_k \cdot c_k \cdot I_k(T_k^{-1}(\vec{x}))}{\sum_{k=1}^N w_k \cdot c_k} $$
///
/// where:
/// - $w_k$ is the user-defined weight for modality $k$
/// - $c_k$ is the data quality/confidence score for modality $k$
/// - $T_k$ is the spatial transformation aligning modality $k$ to the reference coordinate system
///
/// **Theorem (Unbiased Estimator):** If the measurement noise in $I_k$ is zero-mean and uncorrelated,
/// the weighted average is an unbiased estimator of the true tissue property. Minimal variance
/// is achieved when weights are proportional to the inverse variance of the noise (Gauss-Markov theorem).
pub(crate) fn fuse_weighted_average(fusion: &MultiModalFusion) -> KwaversResult<FusedImageResult> {
    let registration = ImageRegistration::default();

    let mut modality_names: Vec<&String> = fusion.registered_data.keys().collect();
    modality_names.sort();

    let reference_name = modality_names.first().ok_or_else(|| {
        KwaversError::Validation(crate::core::error::ValidationError::ConstraintViolation {
            message: "No modalities available for fusion".to_string(),
        })
    })?;

    let reference_modality = fusion.registered_data.get(*reference_name).ok_or_else(|| {
        KwaversError::Validation(crate::core::error::ValidationError::ConstraintViolation {
            message: "Reference modality missing".to_string(),
        })
    })?;

    // Define target grid dimensions based on the reference modality's native grid
    let ref_shape = reference_modality.data.dim();
    let target_dims = (ref_shape.0, ref_shape.1, ref_shape.2);

    let mut fused_intensity = Array3::<f64>::zeros(target_dims);
    let mut confidence_map = Array3::<f64>::zeros(target_dims);
    let mut uncertainty_map = if fusion.config.uncertainty_quantification {
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
            fusion
                .config
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
        let modality = fusion
            .registered_data
            .get(modality_name.as_str())
            .ok_or_else(|| KwaversError::InvalidInput("Modality missing".to_string()))?;

        let weight = fusion
            .config
            .modality_weights
            .get(modality_name.as_str())
            .copied()
            .unwrap_or(1.0);

        modality_quality.insert(modality_name.clone(), modality.quality_score);

        let registration_result = match fusion.config.registration_method {
            crate::physics::acoustics::imaging::fusion::config::RegistrationMethod::RigidBody
            | crate::physics::acoustics::imaging::fusion::config::RegistrationMethod::Automatic => {
                registration.intensity_registration_mutual_info(
                    &reference_modality.data,
                    &modality.data,
                    &identity_transform,
                )?
            }
            crate::physics::acoustics::imaging::fusion::config::RegistrationMethod::Affine
            | crate::physics::acoustics::imaging::fusion::config::RegistrationMethod::NonRigid => {
                return Err(KwaversError::Validation(
                    crate::core::error::ValidationError::ConstraintViolation {
                        message: format!(
                            "Registration method {:?} is not implemented for fusion",
                            fusion.config.registration_method
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
            CowArray::from(modality.data.view())
        } else {
            CowArray::from(registration::resample_to_target_grid(
                &modality.data,
                &registration_result.transform_matrix,
                target_dims,
            ))
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
            fusion.config.output_resolution,
        ),
    })
}
