//! Quality assessment and uncertainty quantification for multi-modal fusion.
//!
//! This module provides methods for evaluating the quality of imaging data from
//! different modalities, computing confidence maps, and quantifying uncertainty
//! in fusion results.

use crate::domain::imaging::ultrasound::elastography::ElasticityMap;
use ndarray::Array3;

struct ImageMetrics {
    snr: f64,
    cnr: f64,
}

fn calculate_image_metrics(data: &Array3<f64>) -> ImageMetrics {
    if data.is_empty() {
        return ImageMetrics { snr: 0.0, cnr: 0.0 };
    }

    let mut values: Vec<f64> = data
        .iter()
        .cloned()
        .filter(|x| !x.is_nan())
        .collect();

    // Sort to determine signal and background regions
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let count = values.len();

    // Define ROI (Region of Interest) and Background
    // Heuristic: Signal is top 10%, Background is bottom 50%
    let signal_threshold_idx = (count as f64 * 0.9).floor() as usize;
    let background_threshold_idx = (count as f64 * 0.5).floor() as usize;

    // Handle edge case where image is too small or flat
    if count < 10 || signal_threshold_idx >= count || background_threshold_idx == 0 {
        // Fallback to simple mean/std
        let mean: f64 = values.iter().sum::<f64>() / count as f64;
        let variance: f64 = values
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / count as f64;
        let std_dev = variance.sqrt();
        let snr = if std_dev > 0.0 { mean / std_dev } else { 0.0 };
        return ImageMetrics { snr, cnr: 0.0 };
    }

    let signal_values = &values[signal_threshold_idx..];
    let background_values = &values[..background_threshold_idx];

    let signal_mean: f64 = signal_values.iter().sum::<f64>() / signal_values.len() as f64;

    let bg_mean: f64 = background_values.iter().sum::<f64>() / background_values.len() as f64;
    let bg_variance: f64 = background_values
        .iter()
        .map(|x| (x - bg_mean).powi(2))
        .sum::<f64>()
        / background_values.len() as f64;
    let bg_std = bg_variance.sqrt();

    let snr = if bg_std > 0.0 {
        signal_mean / bg_std
    } else {
        0.0 // Avoid division by zero
    };

    let cnr = if bg_std > 0.0 {
        (signal_mean - bg_mean).abs() / bg_std
    } else {
        0.0
    };

    ImageMetrics { snr, cnr }
}

/// Compute photoacoustic image quality score
///
/// Evaluates the quality of photoacoustic imaging data based on
/// signal strength and artifact analysis.
///
/// # Arguments
///
/// * `pa_result` - Photoacoustic imaging result
///
/// # Returns
///
/// Quality score in range [0, 1]
pub fn compute_pa_quality(reconstructed_image: &Array3<f64>) -> f64 {
    let metrics = calculate_image_metrics(reconstructed_image);

    // Score components
    // SNR: Expecting > 20 for good quality. Linear saturation.
    let snr_score = (metrics.snr / 20.0).min(1.0);

    // CNR: Expecting > 5 for good contrast.
    let cnr_score = (metrics.cnr / 5.0).min(1.0);

    // Weighted combination
    0.6 * snr_score + 0.4 * cnr_score
}

/// Compute elastography image quality score
///
/// Evaluates the quality of elastography data based on strain
/// accuracy and signal-to-noise ratio.
///
/// # Arguments
///
/// * `elasticity_map` - Elastography result containing stiffness measurements
///
/// # Returns
///
/// Quality score in range [0, 1]
pub fn compute_elastography_quality(elasticity_map: &ElasticityMap) -> f64 {
    // Elastography quality depends heavily on the stiffness contrast
    let metrics = calculate_image_metrics(&elasticity_map.youngs_modulus);

    // SNR is less critical than CNR in elastography (contrast is key)
    let snr_score = (metrics.snr / 15.0).min(1.0);
    let cnr_score = (metrics.cnr / 4.0).min(1.0);

    // Add a sanity check for physically realistic values (e.g. non-negative)
    let (min_e, _max_e, _mean_e) = elasticity_map.statistics();
    let realism_score = if min_e >= 0.0 { 1.0 } else { 0.5 };

    0.3 * snr_score + 0.5 * cnr_score + 0.2 * realism_score
}

/// Compute optical image quality score
///
/// Evaluates the quality of optical imaging data based on intensity
/// statistics and wavelength characteristics.
///
/// # Arguments
///
/// * `optical_intensity` - 3D optical intensity data
/// * `wavelength` - Optical wavelength in meters
///
/// # Returns
///
/// Quality score in range [0, 1]
pub fn compute_optical_quality(optical_intensity: &Array3<f64>, wavelength: f64) -> f64 {
    // Compute basic intensity statistics
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
        1.0 // Visible light: optimal
    } else {
        0.8 // Non-visible: reduced quality factor
    };

    let snr_factor = (snr / 10.0).min(1.0); // Normalize SNR to [0, 1]

    // Composite quality: base quality + wavelength factor + SNR factor
    0.6 + 0.3 * wavelength_factor + 0.1 * snr_factor
}

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

/// Estimate noise characteristics for a modality
///
/// Analyzes the noise properties of imaging data to inform fusion weighting.
///
/// # Arguments
///
/// * `data` - 3D imaging data
///
/// # Returns
///
/// Estimated noise variance
pub fn estimate_modality_noise(data: &Array3<f64>) -> f64 {
    // Compute local variance as noise estimate
    // In practice, would use more sophisticated methods like:
    // - Wavelet-based noise estimation
    // - Background region analysis
    // - Temporal variance analysis

    let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
    let variance: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;

    variance
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_image_metrics() {
        // Create a simple image with signal and background
        // Background: 10 +/- 1
        // Signal: 50
        let mut data = Array3::<f64>::from_elem((10, 10, 1), 10.0);
        // Add some noise to background (bottom 50% of values)
        // Ensure these indices don't overlap with signal (diagonal)
        // Diagonal is (0,0), (1,1)...(9,9).
        // (0,1) and (1,0) are safe.
        data[[0, 1, 0]] = 9.0;
        data[[1, 0, 0]] = 11.0;

        // Add signal (top 10% = 10 pixels) on diagonal
        for i in 0..10 {
            data[[i, i, 0]] = 50.0;
        }

        let metrics = calculate_image_metrics(&data);

        // Signal mean should be 50.0
        // Background mean approx 10.0
        // Background std approx > 0 because of 9.0 and 11.0

        assert!(metrics.snr > 1.0);
        assert!(metrics.cnr > 1.0);
    }

    #[test]
    fn test_compute_pa_quality() {
        let mut data = Array3::<f64>::from_elem((10, 10, 1), 1.0);
        // Add background noise to ensure non-zero std dev
        data[[0, 1, 0]] = 0.9;
        data[[1, 0, 0]] = 1.1;

        // High quality signal
        data[[5, 5, 0]] = 100.0; // Peak

        let quality = compute_pa_quality(&data);
        assert!(quality > 0.0 && quality <= 1.0);

        // Compare with flat image (zero variance -> zero SNR/CNR -> quality 0.0)
        let flat_data = Array3::<f64>::from_elem((10, 10, 1), 1.0);
        let flat_quality = compute_pa_quality(&flat_data);
        assert!(quality > flat_quality);
    }

    #[test]
    fn test_compute_elastography_quality() {
        let youngs = Array3::<f64>::from_elem((10, 10, 1), 1000.0);
        let shear = Array3::<f64>::zeros((10, 10, 1));
        let speed = Array3::<f64>::zeros((10, 10, 1));

        // Add noise to background
        let mut youngs = youngs;
        youngs[[0, 1, 0]] = 900.0;
        youngs[[1, 0, 0]] = 1100.0;

        let mut map = ElasticityMap {
            youngs_modulus: youngs,
            shear_modulus: shear,
            shear_wave_speed: speed,
        };

        // Add stiff inclusion
        map.youngs_modulus[[5, 5, 0]] = 5000.0;

        let quality = compute_elastography_quality(&map);
        assert!(quality > 0.0 && quality <= 1.0);

        // Test negative values penalty
        // We need to keep the background noise/signal structure to keep SNR/CNR similar
        // but introduce a negative value to trigger the penalty.
        let mut bad_map = map.clone();
        bad_map.youngs_modulus[[9, 0, 0]] = -100.0;
        // Note: -100.0 will likely be in the background (lowest value),
        // increasing background variance, which reduces SNR and CNR.
        // Combined with the realism penalty, the score should drop significantly.

        let bad_quality = compute_elastography_quality(&bad_map);
        assert!(bad_quality < quality);
    }

    #[test]
    fn test_compute_optical_quality_visible_light() {
        let intensity = Array3::<f64>::from_elem((8, 8, 4), 100.0);
        let wavelength = 550e-9; // Green light (visible)

        let quality = compute_optical_quality(&intensity, wavelength);

        assert!(quality >= 0.0 && quality <= 1.0);
        assert!(quality > 0.6); // Should be reasonably high
    }

    #[test]
    fn test_compute_optical_quality_infrared() {
        let intensity = Array3::<f64>::from_elem((8, 8, 4), 100.0);
        let wavelength = 1000e-9; // Infrared (non-visible)

        let quality = compute_optical_quality(&intensity, wavelength);

        assert!(quality >= 0.0 && quality <= 1.0);
    }

    #[test]
    fn test_compute_optical_quality_zero_variance() {
        let intensity = Array3::<f64>::from_elem((8, 8, 4), 100.0);
        let wavelength = 550e-9;

        let quality = compute_optical_quality(&intensity, wavelength);

        // Zero variance = zero SNR, but still gets base + wavelength factor
        assert!(quality >= 0.6);
    }

    #[test]
    fn test_estimate_modality_noise() {
        let mut data = Array3::<f64>::from_elem((8, 8, 4), 100.0);
        data[[0, 0, 0]] = 110.0;
        data[[1, 1, 1]] = 90.0;

        let noise = estimate_modality_noise(&data);

        assert!(noise >= 0.0);
        assert!(noise > 0.0); // Should detect variance
    }

    #[test]
    fn test_bayesian_fusion_single_voxel_empty() {
        let (mean, uncertainty) = bayesian_fusion_single_voxel(&[], &[]);
        assert_eq!(mean, 0.0);
        assert_eq!(uncertainty, 1.0); // Maximum uncertainty
    }

    #[test]
    fn test_bayesian_fusion_single_voxel_single_value() {
        let (mean, uncertainty) = bayesian_fusion_single_voxel(&[5.0], &[0.8]);
        assert_eq!(mean, 5.0);
        assert!((uncertainty - 0.2).abs() < 1e-10); // 1 - weight
    }

    #[test]
    fn test_bayesian_fusion_single_voxel_multiple_values() {
        let values = vec![1.0, 2.0, 3.0];
        let weights = vec![1.0, 1.0, 1.0];

        let (mean, uncertainty) = bayesian_fusion_single_voxel(&values, &weights);

        assert!((mean - 2.0).abs() < 1e-10); // Mean of 1, 2, 3
        assert!(uncertainty >= 0.0 && uncertainty <= 1.0);
        assert!(uncertainty > 0.0); // Should have some uncertainty due to variance
    }

    #[test]
    fn test_bayesian_fusion_weighted_average() {
        let values = vec![1.0, 3.0];
        let weights = vec![0.75, 0.25];

        let (mean, _uncertainty) = bayesian_fusion_single_voxel(&values, &weights);

        let expected = (1.0 * 0.75 + 3.0 * 0.25) / (0.75 + 0.25);
        assert!((mean - expected).abs() < 1e-10);
    }

    #[test]
    fn test_compute_fusion_uncertainty() {
        let data1 = Array3::<f64>::from_elem((4, 4, 2), 1.0);
        let data2 = Array3::<f64>::from_elem((4, 4, 2), 3.0);
        let data3 = Array3::<f64>::from_elem((4, 4, 2), 2.0);

        let modality_data = vec![&data1, &data2, &data3];
        let weights = vec![1.0, 1.0, 1.0];

        let uncertainty = compute_fusion_uncertainty(&modality_data, &weights);

        assert_eq!(uncertainty.dim(), (4, 4, 2));

        // All voxels should have same uncertainty (uniform data)
        let first_uncertainty = uncertainty[[0, 0, 0]];
        for value in uncertainty.iter() {
            assert!((value - first_uncertainty).abs() < 1e-10);
        }

        assert!(first_uncertainty > 0.0); // Should have uncertainty due to variance
    }

    #[test]
    fn test_compute_fusion_uncertainty_empty() {
        let uncertainty = compute_fusion_uncertainty(&[], &[]);
        assert_eq!(uncertainty.dim(), (1, 1, 1));
        assert_eq!(uncertainty[[0, 0, 0]], 1.0); // Maximum uncertainty
    }

    #[test]
    fn test_compute_confidence_map() {
        let quality_scores = vec![0.8, 0.9, 0.7];
        let uncertainty = Array3::<f64>::from_elem((4, 4, 2), 0.2);

        let confidence = compute_confidence_map(&quality_scores, &uncertainty);

        assert_eq!(confidence.dim(), (4, 4, 2));

        let avg_quality = (0.8 + 0.9 + 0.7) / 3.0;
        let expected_confidence = avg_quality * (1.0 - 0.2);

        for value in confidence.iter() {
            assert!((value - expected_confidence).abs() < 1e-10);
        }
    }

    #[test]
    fn test_compute_confidence_map_no_quality_scores() {
        let uncertainty = Array3::<f64>::from_elem((4, 4, 2), 0.3);

        let confidence = compute_confidence_map(&[], &uncertainty);

        // Should use default quality of 0.5
        let expected_confidence = 0.5 * (1.0 - 0.3);

        for value in confidence.iter() {
            assert!((value - expected_confidence).abs() < 1e-10);
        }
    }
}
