use super::MultiModalFusion;
use crate::core::error::{KwaversError, KwaversResult};
use crate::physics::acoustics::imaging::fusion::quality;
use crate::physics::acoustics::imaging::fusion::registration;
use crate::physics::acoustics::imaging::fusion::types::{AffineTransform, FusedImageResult};
use crate::physics::imaging::registration::ImageRegistration;
use ndarray::{Array3, CowArray};
use std::collections::HashMap;

/// Probabilistic fusion with uncertainty modeling
///
/// Performs Bayesian fusion that accounts for uncertainty in each modality.
/// This method provides uncertainty estimates at each voxel based on the
/// consistency and quality of the multi-modal measurements.
///
/// # Mathematical Specification
///
/// Let $P(F(\vec{x}) | I_1(\vec{x}), \dots, I_N(\vec{x}))$ be the posterior probability of the fused
/// tissue property $F$ given independent measurements $I_k$ from $N$ modalities.
///
/// By Bayes' Theorem:
/// $$ P(F | I_{1..N}) \propto P(F) \prod_{k=1}^N P(I_k | F) $$
///
/// Assuming a Gaussian noise model $I_k \sim \mathcal{N}(F, \sigma_k^2)$ and an uninformative prior $P(F)$,
/// the maximum a posteriori (MAP) estimate for $F$ is the precision-weighted mean:
///
/// $$ F_{MAP}(\vec{x}) = \frac{\sum_{k=1}^N \tau_k I_k(\vec{x})}{\sum_{k=1}^N \tau_k} $$
///
/// where $\tau_k = (\sigma_k^2)^{-1}$ is the precision (confidence) of modality $k$.
/// The resulting uncertainty corresponds to the Cramér-Rao lower bound for the variance of the estimator:
/// $$ \text{Var}(F_{MAP}) = \left( \sum_{k=1}^N \tau_k \right)^{-1} $$
pub(crate) fn fuse_probabilistic(fusion: &MultiModalFusion) -> KwaversResult<FusedImageResult> {
    let registration = ImageRegistration::default();

    // Use first modality as reference
    let reference_modality = fusion.registered_data.values().next().ok_or_else(|| {
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

    for (modality_name, modality) in &fusion.registered_data {
        let weight = fusion
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
            fusion.config.output_resolution,
        ),
    })
}
