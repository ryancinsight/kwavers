//! Principal-component multimodal image fusion.
//!
//! # Theorem: Maximum-variance loading
//!
//! Let `X in R^(N x M)` contain `N` registered voxels and `M` centered
//! modality intensities. The first principal loading vector `v` solves
//! `argmax_{||v||_2 = 1} v^T C v`, where `C = X^T X / (N - 1)`.
//! The Rayleigh-Ritz theorem gives this maximum at an eigenvector associated
//! with the largest eigenvalue of the symmetric positive semidefinite
//! covariance matrix `C`.
//!
//! # Proof sketch
//!
//! Since `C` is real symmetric, it has an orthonormal eigenbasis. Expanding
//! any unit vector in that basis makes `v^T C v` a convex combination of the
//! eigenvalues, so the maximum is the largest eigenvalue. The implementation
//! computes that loading by power iteration and normalizes absolute loadings
//! into convex weights. Therefore each fused voxel is a weighted sum of the
//! registered modality measurements and lies between their pointwise minimum
//! and maximum.
//!
//! References: Jolliffe (2002) Principal Component Analysis; `A Review of
//! Multimodal Medical Image Fusion Techniques`, Biomed Res Int. 2020.

use super::MultiModalFusion;
use crate::acoustics::imaging::fusion::registration::generate_coordinate_arrays;
use crate::acoustics::imaging::fusion::types::{FusedImageResult, RegisteredModality};
use kwavers_core::error::{KwaversError, KwaversResult};
use ndarray::Array3;
use std::collections::HashMap;

const POWER_ITERATIONS: usize = 128;
const POWER_TOLERANCE: f64 = 1e-12;
/// Fuse pca.
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
///
pub(crate) fn fuse_pca(fusion: &MultiModalFusion) -> KwaversResult<FusedImageResult> {
    let modalities = super::utils::sorted_modalities(fusion)?;
    let dims = super::utils::common_registered_dims(&modalities, "PCA fusion")?;
    let weights = principal_component_weights(&modalities, dims)?;

    let mut intensity_image = Array3::<f64>::zeros(dims);
    let confidence = weighted_quality(&weights, &modalities);
    let mut confidence_map = Array3::<f64>::from_elem(dims, confidence);
    let uncertainty_map = fusion
        .config
        .uncertainty_quantification
        .then(|| Array3::<f64>::from_elem(dims, 1.0 - confidence));

    for ((i, j, k), output) in intensity_image.indexed_iter_mut() {
        *output = weights
            .iter()
            .zip(modalities.iter())
            .map(|(weight, (_, modality))| weight * modality.data[[i, j, k]])
            .sum();
        confidence_map[[i, j, k]] = confidence;
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

fn principal_component_weights(
    modalities: &[(&str, &RegisteredModality)],
    dims: (usize, usize, usize),
) -> KwaversResult<Vec<f64>> {
    let covariance = covariance_matrix(modalities, dims)?;
    Ok(normalized_absolute_principal_loading(
        &covariance,
        modalities.len(),
    ))
}

fn covariance_matrix(
    modalities: &[(&str, &RegisteredModality)],
    dims: (usize, usize, usize),
) -> KwaversResult<Vec<f64>> {
    let modality_count = modalities.len();
    let voxel_count = dims.0 * dims.1 * dims.2;
    let mut means = vec![0.0; modality_count];

    for (index, (name, modality)) in modalities.iter().enumerate() {
        let mut sum = 0.0;
        for value in &modality.data {
            if !value.is_finite() {
                return Err(KwaversError::InvalidInput(format!(
                    "PCA fusion requires finite intensity values; modality {name} contains {value}"
                )));
            }
            sum += value;
        }
        means[index] = sum / voxel_count as f64;
    }

    let denominator = voxel_count.saturating_sub(1).max(1) as f64;
    let mut covariance = vec![0.0; modality_count * modality_count];

    for a in 0..modality_count {
        for b in a..modality_count {
            let mut sum = 0.0;
            for (a_value, b_value) in modalities[a].1.data.iter().zip(modalities[b].1.data.iter()) {
                sum += (a_value - means[a]) * (b_value - means[b]);
            }
            let value = sum / denominator;
            covariance[a * modality_count + b] = value;
            covariance[b * modality_count + a] = value;
        }
    }

    Ok(covariance)
}

fn normalized_absolute_principal_loading(covariance: &[f64], size: usize) -> Vec<f64> {
    let mut vector = vec![1.0 / (size as f64).sqrt(); size];

    for _ in 0..POWER_ITERATIONS {
        let mut next = vec![0.0; size];
        for row in 0..size {
            next[row] = (0..size)
                .map(|col| covariance[row * size + col] * vector[col])
                .sum();
        }

        let norm = next.iter().map(|value| value * value).sum::<f64>().sqrt();
        if norm <= f64::EPSILON {
            return equal_weights(size);
        }
        for value in &mut next {
            *value /= norm;
        }

        let delta = next
            .iter()
            .zip(vector.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        vector = next;
        if delta <= POWER_TOLERANCE {
            break;
        }
    }

    let sum_abs = vector.iter().map(|value| value.abs()).sum::<f64>();
    if sum_abs <= f64::EPSILON {
        return equal_weights(size);
    }

    vector
        .into_iter()
        .map(|value| value.abs() / sum_abs)
        .collect()
}

fn equal_weights(size: usize) -> Vec<f64> {
    vec![1.0 / size as f64; size]
}

fn weighted_quality(weights: &[f64], modalities: &[(&str, &RegisteredModality)]) -> f64 {
    weights
        .iter()
        .zip(modalities.iter())
        .map(|(weight, (_, modality))| weight * modality.quality_score)
        .sum()
}
