//! Ultrasound imaging modalities
//!
//! Implements B-mode, Doppler, elastography, synthetic aperture, plane wave, and coded excitation imaging

use ndarray::Array2;
use num_complex::Complex;

pub mod advanced;
pub mod hifu;

/// Ultrasound imaging mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UltrasoundMode {
    /// Brightness mode (grayscale)
    BMode,
    /// Doppler flow imaging
    Doppler,
    /// Tissue elasticity imaging
    Elastography,
    /// Harmonic imaging
    Harmonic,
}

/// Ultrasound imaging configuration
#[derive(Debug, Clone)]
pub struct UltrasoundConfig {
    /// Imaging mode
    pub mode: UltrasoundMode,
    /// Center frequency (Hz)
    pub frequency: f64,
    /// Sampling frequency (Hz)
    pub sampling_frequency: f64,
    /// Dynamic range (dB)
    pub dynamic_range: f64,
    /// Time gain compensation
    pub tgc_enabled: bool,
}

impl Default for UltrasoundConfig {
    fn default() -> Self {
        Self {
            mode: UltrasoundMode::BMode,
            frequency: 5e6,
            sampling_frequency: 40e6,
            dynamic_range: 60.0,
            tgc_enabled: true,
        }
    }
}

/// Compute B-mode image from RF data
#[must_use]
pub fn compute_bmode_image(rf_data: &Array2<f64>, config: &UltrasoundConfig) -> Array2<f64> {
    let (n_samples, n_lines) = rf_data.dim();
    let mut image = Array2::zeros((n_samples, n_lines));

    for line_idx in 0..n_lines {
        let rf_line = rf_data.column(line_idx);
        let envelope = compute_envelope(&rf_line.to_owned());
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

/// Compute envelope using Hilbert transform approximation
fn compute_envelope(signal: &ndarray::Array1<f64>) -> ndarray::Array1<f64> {
    let n = signal.len();
    let mut envelope = ndarray::Array1::zeros(n);
    for i in 0..n {
        let real = signal[i];
        let imag = if i > 0 && i < n - 1 {
            (signal[i + 1] - signal[i - 1]) / 2.0
        } else {
            0.0
        };
        envelope[i] = (real * real + imag * imag).sqrt();
    }
    envelope
}

const TISSUE_ATTENUATION_COEFFICIENT: f64 = 0.5; // dB/cm/MHz
const SOUND_SPEED_TISSUE: f64 = 1540.0; // m/s
const DB_TO_NEPER: f64 = 8.686; // Conversion factor
const CM_TO_M: f64 = 0.01;
const MHZ_TO_HZ: f64 = 1e6;
const MAX_TGC_GAIN: f64 = 100.0;

fn apply_tgc(signal: &ndarray::Array1<f64>, frequency: f64) -> ndarray::Array1<f64> {
    const DEFAULT_SAMPLING_FREQUENCY: f64 = 40e6;
    apply_tgc_with_sampling(signal, frequency, DEFAULT_SAMPLING_FREQUENCY)
}

fn apply_tgc_with_sampling(
    signal: &ndarray::Array1<f64>,
    frequency: f64,
    sampling_frequency: f64,
) -> ndarray::Array1<f64> {
    let n = signal.len();
    let mut compensated = signal.clone();
    let alpha_np =
        TISSUE_ATTENUATION_COEFFICIENT * frequency / MHZ_TO_HZ * (10.0 * CM_TO_M) / DB_TO_NEPER;
    for i in 0..n {
        let depth = i as f64 * SOUND_SPEED_TISSUE / (2.0 * sampling_frequency);
        let gain = (2.0 * alpha_np * depth).exp();
        compensated[i] *= gain.min(MAX_TGC_GAIN);
    }
    compensated
}

#[must_use]
pub fn compute_doppler_shift(iq_data: &Array2<Complex<f64>>, prf: f64) -> Array2<f64> {
    let (n_samples, n_pulses) = iq_data.dim();
    let mut doppler = Array2::zeros((n_samples, n_pulses - 1));
    for i in 0..n_samples {
        for j in 0..n_pulses - 1 {
            let phase_diff = (iq_data[[i, j + 1]] / iq_data[[i, j]]).arg();
            doppler[[i, j]] = phase_diff * prf / (2.0 * std::f64::consts::PI);
        }
    }
    doppler
}

#[must_use]
pub fn compute_strain(displacement: &Array2<f64>, spatial_resolution: f64) -> Array2<f64> {
    let (n_depth, n_lines) = displacement.dim();
    let mut strain = Array2::zeros((n_depth - 1, n_lines));
    for line in 0..n_lines {
        for depth in 0..n_depth - 1 {
            let gradient = (displacement[[depth + 1, line]] - displacement[[depth, line]])
                / spatial_resolution;
            strain[[depth, line]] = gradient;
        }
    }
    strain
}
