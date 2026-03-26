//! Base image metrics and noise estimation for quality assessment

use ndarray::Array3;

/// Basic image metrics for quality assessment
#[derive(Debug, Clone, Copy)]
pub struct ImageMetrics {
    /// Signal-to-Noise Ratio
    pub snr: f64,
    /// Contrast-to-Noise Ratio
    pub cnr: f64,
}

/// Calculate basic SNR and CNR metrics from 3D image data
#[must_use]
pub fn calculate_image_metrics(data: &Array3<f64>) -> ImageMetrics {
    if data.is_empty() {
        return ImageMetrics { snr: 0.0, cnr: 0.0 };
    }

    let mut values: Vec<f64> = data.iter().copied().filter(|x| !x.is_nan()).collect();

    // Sort to determine signal and background regions
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let count = values.len();

    // Define ROI (Region of Interest) and Background
    // Heuristic: Signal is top 10%, Background is bottom 50%
    let signal_threshold_idx = (count as f64 * 0.9).floor() as usize;
    let background_threshold_idx = (count as f64 * 0.5).floor() as usize;

    // Handle edge case where image is too small or flat
    if count < 10 || signal_threshold_idx >= count || background_threshold_idx == 0 {
        let mean: f64 = values.iter().sum::<f64>() / count as f64;
        let variance: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / count as f64;
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
#[must_use]
pub fn estimate_modality_noise(data: &Array3<f64>) -> f64 {
    // Compute local variance as noise estimate
    let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
    let variance: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;

    variance
}
