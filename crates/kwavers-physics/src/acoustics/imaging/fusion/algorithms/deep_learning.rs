//! Parameter-free attention fusion for the `DeepFusion` strategy.
//!
//! # Theorem: Convex voxel-attention fusion
//!
//! Let registered modality intensities at voxel `x` be `I_m(x)`, modality
//! quality priors be `q_m > 0`, user weights be `w_m > 0`, and normalized
//! salience be `s_m(x) in [0, 1]`. Define
//! `a_m(x) = q_m w_m exp(s_m(x)) / sum_j q_j w_j exp(s_j(x))`.
//! Then `a_m(x) >= 0` and `sum_m a_m(x)=1`, so
//! `F(x)=sum_m a_m(x) I_m(x)` lies in the closed interval spanned by the
//! registered modality values at that voxel.
//!
//! # Proof sketch
//!
//! The exponential and positive priors make each unnormalized attention score
//! non-negative. Normalization by their positive sum creates a probability
//! simplex. `F(x)` is therefore a convex combination of input measurements,
//! and the interval bound follows from elementary convexity.
//!
//! This is not a fabricated trained network. It is the deterministic
//! single-head attention limit used when no validated trained fusion model is
//! present in the public configuration. It follows the same late-fusion
//! attention principle discussed in multimodal medical-image fusion reviews
//! while keeping all weights derived from input data and quality metadata.

use super::MultiModalFusion;
use crate::acoustics::imaging::fusion::registration::generate_coordinate_arrays;
use crate::acoustics::imaging::fusion::types::{FusedImageResult, RegisteredModality};
use kwavers_core::error::{KwaversError, KwaversResult};
use ndarray::Array3;
use std::collections::HashMap;

const MIN_PRIOR: f64 = 1.0e-12;
/// Fuse deep learning.
/// # Errors
/// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
/// - Propagates any [`KwaversError`] returned by called functions.
///
pub(crate) fn fuse_deep_learning(fusion: &MultiModalFusion) -> KwaversResult<FusedImageResult> {
    let modalities = super::utils::sorted_modalities(fusion)?;
    let dims = super::utils::common_registered_dims(&modalities, "DeepFusion attention")?;
    let bounds = normalization_bounds(&modalities)?;

    let mut intensity_image = Array3::<f64>::zeros(dims);
    let mut confidence_map = Array3::<f64>::zeros(dims);
    let mut uncertainty_map = fusion
        .config
        .uncertainty_quantification
        .then(|| Array3::<f64>::zeros(dims));

    for i in 0..dims.0 {
        for j in 0..dims.1 {
            for k in 0..dims.2 {
                let mut denominator = 0.0;
                let mut weighted_value = 0.0;
                let mut weighted_quality = 0.0;
                let mut weights = vec![0.0; modalities.len()];

                for (index, (name, modality)) in modalities.iter().enumerate() {
                    let value = modality.data[[i, j, k]];
                    if !value.is_finite() {
                        return Err(KwaversError::InvalidInput(format!(
                            "DeepFusion attention requires finite intensity values; modality {name} contains {value}"
                        )));
                    }
                    let salience = normalized_salience(value, bounds[index]);
                    let prior = positive_prior(fusion, name, modality)?;
                    let score = prior * salience.exp();
                    weights[index] = score;
                    denominator += score;
                }

                if denominator <= 0.0 || !denominator.is_finite() {
                    return Err(KwaversError::InvalidInput(
                        "DeepFusion attention produced an invalid attention normalization"
                            .to_owned(),
                    ));
                }

                for (score, (_, modality)) in weights.iter().zip(modalities.iter()) {
                    let attention = score / denominator;
                    let value = modality.data[[i, j, k]];
                    weighted_value += attention * value;
                    weighted_quality += attention * modality.quality_score;
                }

                intensity_image[[i, j, k]] = weighted_value;
                confidence_map[[i, j, k]] = weighted_quality.clamp(0.0, 1.0);
                if let Some(ref mut uncertainty) = uncertainty_map {
                    uncertainty[[i, j, k]] = attention_uncertainty(&weights, denominator);
                }
            }
        }
    }

    Ok(FusedImageResult {
        intensity_image,
        tissue_properties: HashMap::new(),
        confidence_map,
        uncertainty_map,
        registration_transforms: super::utils::identity_registration_transforms(&modalities),
        modality_quality: super::utils::modality_quality_map(&modalities),
        coordinates: generate_coordinate_arrays(dims, fusion.config.output_resolution),
    })
}

fn normalization_bounds(
    modalities: &[(&str, &RegisteredModality)],
) -> KwaversResult<Vec<(f64, f64)>> {
    modalities
        .iter()
        .map(|(name, modality)| {
            if modality.data.iter().any(|value| !value.is_finite()) {
                return Err(KwaversError::InvalidInput(format!(
                    "DeepFusion attention requires finite intensity values; modality {name} contains non-finite data"
                )));
            }
            Ok(super::utils::compute_robust_bounds(modality.data.view()))
        })
        .collect()
}

fn positive_prior(
    fusion: &MultiModalFusion,
    name: &str,
    modality: &RegisteredModality,
) -> KwaversResult<f64> {
    if !modality.quality_score.is_finite() {
        return Err(KwaversError::InvalidInput(format!(
            "DeepFusion attention requires finite quality scores; modality {name} has {}",
            modality.quality_score
        )));
    }

    let config_weight = fusion
        .config
        .modality_weights
        .get(name)
        .copied()
        .unwrap_or(1.0);
    if !config_weight.is_finite() || config_weight < 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "DeepFusion attention requires finite non-negative config weights; modality {name} has {config_weight}"
        )));
    }

    Ok((modality.quality_score.max(0.0) * config_weight).max(MIN_PRIOR))
}

fn normalized_salience(value: f64, (lower, upper): (f64, f64)) -> f64 {
    if upper <= lower {
        return 0.0;
    }
    ((value - lower) / (upper - lower)).clamp(0.0, 1.0)
}

fn attention_uncertainty(scores: &[f64], denominator: f64) -> f64 {
    if scores.len() <= 1 {
        return 0.0;
    }

    let entropy = scores
        .iter()
        .map(|score| score / denominator)
        .filter(|probability| *probability > 0.0)
        .map(|probability| -probability * probability.ln())
        .sum::<f64>();
    entropy / (scores.len() as f64).ln()
}
