use super::MultiModalFusion;
use crate::acoustics::imaging::fusion::quality;
use crate::acoustics::imaging::fusion::registration::{self, RitkRegistrationEngine};
use crate::acoustics::imaging::fusion::types::{AffineTransform, FusedImageResult};
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3;
use std::{borrow::Cow, collections::HashMap};

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
/// # Errors
/// - Propagates any `KwaversError` returned by called functions.
///
pub(crate) fn fuse_probabilistic(fusion: &MultiModalFusion) -> KwaversResult<FusedImageResult> {
    let registration_engine = RitkRegistrationEngine::default();

    // Use first modality as reference
    let reference_modality = fusion.registered_data.values().next().ok_or_else(|| {
        KwaversError::Validation(kwavers_core::error::ValidationError::ConstraintViolation {
            message: "No modalities available for fusion".to_owned(),
        })
    })?;

    // Define target grid dimensions based on the reference modality's native grid
    let target_dims = reference_modality.data.shape();

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
        let identity_transform = registration::IDENTITY_HOMOGENEOUS;

        let registration_result = registration_engine.register_for_method(
            &reference_modality.data,
            &modality.data,
            kwavers_imaging::fusion::RegistrationMethod::RigidBody,
        )?;

        let affine_transform =
            AffineTransform::from_homogeneous(&registration_result.transform_matrix);
        registration_transforms.insert(modality_name.clone(), affine_transform);

        // Resample to common grid
        let transform = registration_result.transform_matrix;
        let resampled_data =
            if modality.data.shape() == target_dims && transform == identity_transform {
                Cow::Borrowed(&modality.data)
            } else {
                Cow::Owned(registration_engine.resample_registered(
                    &modality.data,
                    &registration_result,
                    target_dims,
                )?)
            };

        modality_data.push((
            resampled_data,
            weight,
            modality.quality_score,
            registration_result.confidence,
        ));
    }

    // Perform probabilistic fusion at each voxel
    for i in 0..target_dims[0] {
        for j in 0..target_dims[1] {
            for k in 0..target_dims[2] {
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