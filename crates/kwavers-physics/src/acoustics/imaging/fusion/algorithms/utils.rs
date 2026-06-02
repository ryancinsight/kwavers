use super::MultiModalFusion;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_domain::imaging::fusion::AffineTransform;
use crate::acoustics::imaging::fusion::types::RegisteredModality;
use ndarray::ArrayView3;
use std::collections::HashMap;
/// Sorted modalities.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub(crate) fn sorted_modalities(
    fusion: &MultiModalFusion,
) -> KwaversResult<Vec<(&str, &RegisteredModality)>> {
    let mut modality_names: Vec<&String> = fusion.registered_data.keys().collect();
    modality_names.sort();

    modality_names
        .into_iter()
        .map(|name| {
            fusion
                .registered_data
                .get(name)
                .map(|modality| (name.as_str(), modality))
                .ok_or_else(|| KwaversError::InvalidInput(format!("Modality {name} missing")))
        })
        .collect()
}
/// Common registered dims.
/// # Errors
/// - Returns [`KwaversError::DimensionMismatch`] if the precondition for mismatched array or grid dimensions is violated.
/// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
/// - Propagates any [`KwaversError`] returned by called functions.
///
pub(crate) fn common_registered_dims(
    modalities: &[(&str, &RegisteredModality)],
    algorithm_name: &str,
) -> KwaversResult<(usize, usize, usize)> {
    let (reference_name, reference) = modalities.first().ok_or_else(|| {
        KwaversError::Validation(kwavers_core::error::ValidationError::ConstraintViolation {
            message: format!("{algorithm_name} requires at least one registered modality"),
        })
    })?;
    let dims = reference.data.dim();

    if dims.0 == 0 || dims.1 == 0 || dims.2 == 0 {
        return Err(KwaversError::InvalidInput(format!(
            "{algorithm_name} requires non-empty registered volumes; modality {reference_name} has {dims:?}"
        )));
    }

    for (name, modality) in modalities {
        if modality.data.dim() != dims {
            return Err(KwaversError::DimensionMismatch(format!(
                "{algorithm_name} requires identical registered dimensions; reference {dims:?}, modality {name} has {:?}",
                modality.data.dim()
            )));
        }
    }

    Ok(dims)
}

pub(crate) fn modality_quality_map(
    modalities: &[(&str, &RegisteredModality)],
) -> HashMap<String, f64> {
    modalities
        .iter()
        .map(|(name, modality)| ((*name).to_owned(), modality.quality_score))
        .collect()
}

pub(crate) fn identity_registration_transforms(
    modalities: &[(&str, &RegisteredModality)],
) -> HashMap<String, AffineTransform> {
    modalities
        .iter()
        .map(|(name, _)| ((*name).to_owned(), AffineTransform::identity()))
        .collect()
}

/// Compute robust normalization bounds (1st and 99th percentiles)
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
pub(crate) fn compute_robust_bounds(data: ArrayView3<'_, f64>) -> (f64, f64) {
    let mut values: Vec<f64> = data.iter().copied().filter(|v| v.is_finite()).collect();

    if values.is_empty() {
        return (0.0, 0.0);
    }

    values.sort_by(|a, b| a.total_cmp(b));

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
pub(crate) fn classify_voxel_features(us: f64, pa: f64, stiffness: f64) -> (u8, f64) {
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
