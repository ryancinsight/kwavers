//! Modality-specific quality score computations

use super::metrics::calculate_image_metrics;
use crate::domain::imaging::ultrasound::elastography::ElasticityMap;
use ndarray::Array3;

/// Compute photoacoustic image quality score
///
/// Evaluates the quality of photoacoustic imaging data based on
/// signal strength and artifact analysis.
///
/// # Arguments
///
/// * `reconstructed_image` - Photoacoustic imaging result
///
/// # Returns
///
/// Quality score in range [0, 1]
#[must_use]
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
#[must_use]
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
#[must_use]
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
