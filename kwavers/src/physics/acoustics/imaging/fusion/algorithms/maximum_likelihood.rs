use super::MultiModalFusion;
use crate::core::error::{KwaversError, KwaversResult};
use crate::physics::acoustics::imaging::fusion::registration;
use crate::physics::acoustics::imaging::fusion::types::{AffineTransform, FusedImageResult};
use crate::physics::imaging::registration::ImageRegistration;
use ndarray::{Array3, CowArray, Zip};
use std::collections::HashMap;

/// Maximum likelihood estimation fusion
///
/// Statistical estimation method that maximizes the likelihood of the
/// observed multi-modal data given a noise model for each modality.
///
/// # Mathematical Specification
///
/// Let $\mathbf{I}(\vec{x}) = \{I_1(\vec{x}), \dots, I_N(\vec{x})\}$ be the set of registered observations.
/// The goal is to estimate the true underlying image $F(\vec{x})$ and the modality-specific variances $\sigma_k^2$
/// that maximize the log-likelihood:
///
/// $$ \mathcal{L}(F, \sigma^2 | \mathbf{I}) = - \frac{1}{2} \sum_{\vec{x}} \sum_{k=1}^N \left( \log(2\pi\sigma_k^2) + \frac{(I_k(\vec{x}) - F(\vec{x}))^2}{\sigma_k^2} \right) $$
///
/// We optimize this via the Expectation-Maximization (EM) algorithm:
///
/// **E-Step (Estimate Fused Image):**
/// Given current variance estimates $\sigma_k^{(t)2}$, compute the optimal fusion:
/// $$ F^{(t+1)}(\vec{x}) = \frac{\sum_{k=1}^N I_k(\vec{x}) / \sigma_k^{(t)2}}{\sum_{k=1}^N 1 / \sigma_k^{(t)2}} $$
///
/// **M-Step (Update Variances):**
/// Given the estimated fusion $F^{(t+1)}$, update the variance for each modality:
/// $$ \sigma_k^{(t+1)2} = \frac{1}{|\Omega|} \sum_{\vec{x} \in \Omega} (I_k(\vec{x}) - F^{(t+1)}(\vec{x}))^2 $$
///
/// **Theorem (EM Monotonicity):** Each EM iteration monotonically increases the likelihood $\mathcal{L}$,
/// converging to a local maximum or saddle point of the likelihood function.
pub(crate) fn fuse_maximum_likelihood(
    fusion: &MultiModalFusion,
) -> KwaversResult<FusedImageResult> {
    let image_reg = ImageRegistration::default();

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
    let mut uncertainty_map = Array3::<f64>::zeros(target_dims);

    let mut registration_transforms = HashMap::new();
    let mut modality_quality = HashMap::new();

    // 1. Prepare and register all data
    // We store: (resampled_data, current_variance)
    struct ModalityContext<'a> {
        data: CowArray<'a, f64, ndarray::Ix3>,
        variance: f64,
    }
    let mut contexts = Vec::new();

    let identity_transform = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    for modality_name in &modality_names {
        let modality = fusion.registered_data.get(modality_name.as_str()).unwrap();
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
            let sum_sq_error: f64 = Zip::from(&ctx.data)
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
            fusion.config.output_resolution,
        ),
    })
}
