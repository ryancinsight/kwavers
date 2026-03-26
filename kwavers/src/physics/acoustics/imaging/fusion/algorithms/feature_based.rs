use super::utils::{classify_voxel_features, compute_robust_bounds};
use super::MultiModalFusion;
use crate::core::error::{KwaversError, KwaversResult};
use crate::physics::acoustics::imaging::fusion::registration;
use crate::physics::acoustics::imaging::fusion::types::{AffineTransform, FusedImageResult};
use crate::physics::imaging::registration::ImageRegistration;
use ndarray::{Array3, CowArray};
use std::collections::HashMap;

/// Feature-based fusion using complementary tissue properties
///
/// Advanced fusion using tissue property relationships. For example:
/// PA absorption + US scattering + Elastography stiffness.
///
/// # Mathematical Specification
///
/// Let $\vec{f}(\vec{x}) = [f_1(\vec{x}), f_2(\vec{x}), \dots, f_M(\vec{x})]^T$ be a feature vector extracted
/// from the registered multi-modal imaging data at spatial coordinate $\vec{x}$.
///
/// The tissue classification function $C: \mathbb{R}^M \rightarrow \mathcal{T}$ maps the feature
/// vector to a discrete set of tissue types $\mathcal{T}$ (e.g., fluid, soft tissue, vessel, calcification).
///
/// The feature-based fused intensity $F(\vec{x})$ is defined using adaptive weights $w_k(\vec{f}(\vec{x}))$ that
/// depend on the classified tissue type:
///
/// $$ F(\vec{x}) = \frac{\sum_{k=1}^N w_k(\vec{f}(\vec{x})) \cdot c_k \cdot I_k(\vec{x})}{\sum_{k=1}^N w_k(\vec{f}(\vec{x})) \cdot c_k} $$
///
/// **Theorem (Complementary Information):** If the mutual information $I(I_A; I_B)$ between two
/// modalities is low, they provide complementary features. Feature-based fusion adaptively selects
/// the modality with the highest mutual information with the underlying true tissue class $T$,
/// maximizing $I(F; T) \geq \max_k I(I_k; T)$.
pub(crate) fn fuse_feature_based(fusion: &MultiModalFusion) -> KwaversResult<FusedImageResult> {
    let registration = ImageRegistration::default();

    // 1. Setup Reference and Grid
    // Sort keys to ensure deterministic reference selection
    let mut modality_names: Vec<&String> = fusion.registered_data.keys().collect();
    modality_names.sort();

    let reference_name = modality_names.first().ok_or_else(|| {
        KwaversError::Validation(crate::core::error::ValidationError::ConstraintViolation {
            message: "No modalities available for fusion".to_string(),
        })
    })?;

    let reference_modality = fusion.registered_data.get(*reference_name).ok_or_else(|| {
        KwaversError::InvalidInput("Reference modality missing".to_string())
    })?;

    let ref_shape = reference_modality.data.dim();
    let target_dims = (ref_shape.0, ref_shape.1, ref_shape.2);

    // 2. Register & Resample All Modalities
    struct Channel<'a> {
        name: String,
        data: CowArray<'a, f64, ndarray::Ix3>,
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
        let modality = fusion.registered_data.get(name).unwrap();
        let weight = fusion
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
                CowArray::from(modality.data.view())
            } else {
                CowArray::from(registration::resample_to_target_grid(
                    &modality.data,
                    &reg_result.transform_matrix,
                    target_dims,
                ))
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
        let (min, max) = compute_robust_bounds(ch.data.view());
        let scale = if max > min { 1.0 / (max - min) } else { 0.0 };
        norm_params.push(NormParams { min, scale });
    }

    // 4. Fusion Loop
    let mut fused_intensity = Array3::<f64>::zeros(target_dims);
    let mut confidence_map = Array3::<f64>::zeros(target_dims);
    let mut tissue_class_map = Array3::<f64>::zeros(target_dims);
    let mut classification_confidence_map = Array3::<f64>::zeros(target_dims);

    // Initialize uncertainty map if enabled
    let mut uncertainty_map = if fusion.config.uncertainty_quantification {
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
                    let norm_val =
                        ((val - norm_params[idx].min) * norm_params[idx].scale).clamp(0.0, 1.0);

                    if ch.name.contains("ultrasound") {
                        us_val = norm_val;
                    } else if ch.name.contains("photoacoustic") {
                        pa_val = norm_val;
                    } else if ch.name.contains("elastography") {
                        stiff_val = norm_val;
                    }
                }

                let (tissue_type, class_conf) = classify_voxel_features(us_val, pa_val, stiff_val);
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
                        2
                            // Vessel: Trust PA highly
                            if ch.name.contains("photoacoustic") => {
                                w *= 2.0;
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

                    if uncertainty_map.is_some() {
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
            fusion.config.output_resolution,
        ),
    })
}
