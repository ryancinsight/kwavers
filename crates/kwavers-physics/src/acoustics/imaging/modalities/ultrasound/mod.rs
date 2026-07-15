//! Ultrasound imaging modalities
//!
//! Implements B-mode, Doppler, elastography, synthetic aperture, plane wave, and coded excitation imaging

use eunomia::Complex;
use kwavers_core::constants::acoustic_parameters::NP_TO_DB;
use kwavers_core::constants::fundamental::{ACOUSTIC_ABSORPTION_TISSUE, SOUND_SPEED_TISSUE};
use kwavers_core::constants::numerical::{CM_TO_M, MHZ_TO_HZ, TWO_PI};
use leto::Array2;

pub mod advanced;
pub mod frequency_domain_fwi;
pub mod hifu;

// Re-export domain types for convenience
pub use kwavers_imaging::ultrasound::{UltrasoundConfig, UltrasoundMode};

/// Compute B-mode image from RF data
#[must_use]
pub fn compute_bmode_image(rf_data: &Array2<f64>, config: &UltrasoundConfig) -> Array2<f64> {
    let [n_samples, n_lines] = rf_data.shape();
    let mut image = Array2::zeros([n_samples, n_lines]);

    for line_idx in 0..n_lines {
        let rf_line = rf_data.index_axis(1, line_idx).expect("valid axis index");
        let envelope = compute_envelope(&rf_line.to_contiguous());
        let compensated = if config.tgc_enabled {
            apply_tgc(&envelope, config.frequency)
        } else {
            envelope
        };
        for (i, &value) in compensated.iter().enumerate() {
            let log_value = 20.0 * (value.max(1e-10)).log10();
            let normalized = (log_value + config.dynamic_range) / config.dynamic_range;
            image[[i, line_idx]] = normalized.clamp(0.0, 1.0);
        }
    }
    image
}

/// Compute envelope using Hilbert transform approximation.
///
/// ## Not yet implemented
///
/// - **Analytical Hilbert transform**: FFT-based quadrature for exact envelope detection.
/// - **Advanced wall filtering**: Regression, polynomial, and eigenvector-based clutter
///   suppression for Doppler processing.
/// - **Speckle reduction**: Anisotropic diffusion and wavelet-domain denoising.
/// - **Harmonic imaging**: Tissue harmonic and compound harmonic B-mode reconstruction.
/// - **Spatial compounding**: Multi-angle coherent compounding for artifact reduction.
fn compute_envelope(signal: &leto::Array1<f64>) -> leto::Array1<f64> {
    let n = signal.len();
    let mut envelope = leto::Array1::zeros([n]);
    for i in 0..n {
        let real = signal[i];
        let imag = if i > 0 && i < n - 1 {
            (signal[i + 1] - signal[i - 1]) / 2.0
        } else {
            0.0
        };
        envelope[i] = real.hypot(imag);
    }
    envelope
}

// SSOT: ACOUSTIC_ABSORPTION_TISSUE = 0.5 dB/(cm·MHz) from core::constants::fundamental
// SOUND_SPEED_TISSUE = 1540.0 m/s imported from kwavers_core::constants::fundamental
const MAX_TGC_GAIN: f64 = 100.0;

fn apply_tgc(signal: &leto::Array1<f64>, frequency: f64) -> leto::Array1<f64> {
    const DEFAULT_SAMPLING_FREQUENCY: f64 = 40e6;
    apply_tgc_with_sampling(signal, frequency, DEFAULT_SAMPLING_FREQUENCY)
}

fn apply_tgc_with_sampling(
    signal: &leto::Array1<f64>,
    frequency: f64,
    sampling_frequency: f64,
) -> leto::Array1<f64> {
    let n = signal.len();
    let mut compensated = signal.clone();
    // α[Np/m] = α[dB/(cm·MHz)] × f[MHz] × (1/CM_TO_M) [cm/m] × (1/NP_TO_DB) [Np/dB]
    let alpha_np = ACOUSTIC_ABSORPTION_TISSUE * (frequency / MHZ_TO_HZ) / CM_TO_M / NP_TO_DB;
    for i in 0..n {
        let depth = i as f64 * SOUND_SPEED_TISSUE / (2.0 * sampling_frequency);
        let gain = (2.0 * alpha_np * depth).exp();
        compensated[i] *= gain.min(MAX_TGC_GAIN);
    }
    compensated
}

#[must_use]
pub fn compute_doppler_shift(iq_data: &Array2<Complex<f64>>, prf: f64) -> Array2<f64> {
    let [n_samples, n_pulses] = iq_data.shape();
    let mut doppler = Array2::zeros([n_samples, n_pulses - 1]);
    for i in 0..n_samples {
        for j in 0..n_pulses - 1 {
            let phase_diff = (iq_data[[i, j + 1]] / iq_data[[i, j]]).arg();
            doppler[[i, j]] = phase_diff * prf / (TWO_PI);
        }
    }
    doppler
}

#[must_use]
pub fn compute_strain(displacement: &Array2<f64>, spatial_resolution: f64) -> Array2<f64> {
    let [n_depth, n_lines] = displacement.shape();
    let mut strain = Array2::zeros([n_depth - 1, n_lines]);
    for line in 0..n_lines {
        for depth in 0..n_depth - 1 {
            let gradient = (displacement[[depth + 1, line]] - displacement[[depth, line]])
                / spatial_resolution;
            strain[[depth, line]] = gradient;
        }
    }
    strain
}
