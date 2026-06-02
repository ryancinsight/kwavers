//! Bayesian fusion uncertainty and confidence mapping

use ndarray::Array3;

/// Compute fusion uncertainty using multi-modal variance
///
/// Estimates the uncertainty in fusion results by analyzing the variance
/// and consistency across different modalities.
///
/// # Arguments
///
/// * `modality_data` - Slice of modality data arrays
/// * `weights` - Slice of modality weights
///
/// # Returns
///
/// Uncertainty map (higher values = greater uncertainty)
#[must_use]
pub fn compute_fusion_uncertainty(modality_data: &[&Array3<f64>], weights: &[f64]) -> Array3<f64> {
    if modality_data.is_empty() {
        return Array3::<f64>::ones((1, 1, 1)); // Maximum uncertainty
    }

    let dims = modality_data[0].dim();
    let mut uncertainty_map = Array3::<f64>::zeros(dims);

    // For each voxel, compute weighted variance across modalities
    for i in 0..dims.0 {
        for j in 0..dims.1 {
            for k in 0..dims.2 {
                let values: Vec<f64> = modality_data.iter().map(|data| data[[i, j, k]]).collect();

                let (_mean, uncertainty) = bayesian_fusion_single_voxel(&values, weights);
                uncertainty_map[[i, j, k]] = uncertainty;
            }
        }
    }

    uncertainty_map
}

/// Bayesian fusion for a single voxel with uncertainty quantification
///
/// Performs Bayesian fusion of multiple measurements at a single spatial
/// location, returning both the fused value and its uncertainty.
///
/// # Arguments
///
/// * `values` - Measurements from different modalities
/// * `weights` - Reliability weights for each modality
///
/// # Returns
///
/// Tuple of (fused_value, uncertainty) where uncertainty is in [0, 1]
#[must_use]
pub fn bayesian_fusion_single_voxel(values: &[f64], weights: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 1.0); // High uncertainty for no data
    }

    if values.len() == 1 {
        return (values[0], 1.0 - weights[0].min(1.0)); // Uncertainty inversely proportional to weight
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

/// Compute confidence map from quality scores and uncertainty
///
/// Combines quality metrics and uncertainty estimates to produce a
/// spatial confidence map for the fusion result.
///
/// # Arguments
///
/// * `quality_scores` - Per-modality quality scores
/// * `uncertainty_map` - Spatial uncertainty map
///
/// # Returns
///
/// Confidence map in range [0, 1]
#[must_use]
pub fn compute_confidence_map(
    quality_scores: &[f64],
    uncertainty_map: &Array3<f64>,
) -> Array3<f64> {
    // Average quality across modalities
    let avg_quality = if !quality_scores.is_empty() {
        quality_scores.iter().sum::<f64>() / quality_scores.len() as f64
    } else {
        0.5
    };

    // Confidence = quality * (1 - uncertainty)
    uncertainty_map.mapv(|uncertainty| avg_quality * (1.0 - uncertainty))
}
