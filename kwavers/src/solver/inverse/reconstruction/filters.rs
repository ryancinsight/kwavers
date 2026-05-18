//! Reconstruction filter implementations

use crate::core::error::KwaversResult;
use ndarray::Array2;
use std::f64::consts::PI;

use super::config::ReconstructionFilterType;

/// Apply reconstruction filter to sensor data
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn apply_reconstruction_filter(
    data: &Array2<f64>,
    filter_type: &ReconstructionFilterType,
    sampling_freq: f64,
) -> KwaversResult<Array2<f64>> {
    match filter_type {
        ReconstructionFilterType::None => Ok(data.clone()),
        ReconstructionFilterType::RamLak => apply_ram_lak_filter(data, sampling_freq),
        ReconstructionFilterType::SheppLogan => apply_shepp_logan_filter(data, sampling_freq),
        ReconstructionFilterType::Cosine => apply_cosine_filter(data, sampling_freq),
        ReconstructionFilterType::Hamming => apply_hamming_filter(data, sampling_freq),
        ReconstructionFilterType::Hann => apply_hann_filter(data, sampling_freq),
    }
}

/// Ram-Lak (ramp) filter
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
fn apply_ram_lak_filter(data: &Array2<f64>, sampling_freq: f64) -> KwaversResult<Array2<f64>> {
    let n = data.dim().1;
    let mut filtered = Array2::zeros(data.dim());

    for sensor_idx in 0..data.dim().0 {
        let trace = data.row(sensor_idx).to_owned();
        let filtered_trace = crate::math::fft::apply_spectral_response_1d(
            &trace,
            sampling_freq,
            move |i, freq, nyquist| {
                if i <= n / 2 {
                    freq / nyquist
                } else {
                    (n - i) as f64 * (sampling_freq / n as f64) / nyquist
                }
            },
        );
        filtered.row_mut(sensor_idx).assign(&filtered_trace);
    }

    Ok(filtered)
}

/// Shepp-Logan filter
///
/// Implements the Shepp-Logan filter with frequency response:
/// H(f) = |f| * (2/π) * sin(πf/2f_max)
///
/// The Shepp-Logan filter provides better suppression of high frequencies
/// compared to Ram-Lak while maintaining good resolution.
///
/// # References
/// - Kak & Slaney (1988): "Principles of Computerized Tomographic Imaging", Chapter 3
/// - Shepp & Logan (1974): "The Fourier reconstruction of a head section"
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
fn apply_shepp_logan_filter(data: &Array2<f64>, sampling_freq: f64) -> KwaversResult<Array2<f64>> {
    let mut filtered = Array2::zeros(data.dim());

    for sensor_idx in 0..data.dim().0 {
        let trace = data.row(sensor_idx).to_owned();
        let filtered_trace = crate::math::fft::apply_spectral_response_1d(
            &trace,
            sampling_freq,
            move |_, freq, nyquist| {
                let f_max = nyquist;
                freq.abs() * (2.0 / PI) * (PI * freq / (2.0 * f_max)).sin()
            },
        );
        filtered.row_mut(sensor_idx).assign(&filtered_trace);
    }

    Ok(filtered)
}

/// Cosine filter
///
/// Implements the cosine filter with frequency response:
/// H(f) = |f| * cos(πf/2f_max)
///
/// The cosine filter provides smooth frequency response with gradual
/// high-frequency attenuation, reducing ringing artifacts.
///
/// # References
/// - Kak & Slaney (1988): "Principles of Computerized Tomographic Imaging", Chapter 3
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
fn apply_cosine_filter(data: &Array2<f64>, sampling_freq: f64) -> KwaversResult<Array2<f64>> {
    let mut filtered = Array2::zeros(data.dim());

    for sensor_idx in 0..data.dim().0 {
        let trace = data.row(sensor_idx).to_owned();
        let filtered_trace = crate::math::fft::apply_spectral_response_1d(
            &trace,
            sampling_freq,
            move |_, freq, nyquist| {
                let f_max = nyquist;
                freq.abs() * (PI * freq / (2.0 * f_max)).cos()
            },
        );
        filtered.row_mut(sensor_idx).assign(&filtered_trace);
    }

    Ok(filtered)
}

/// Hamming window filter
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
fn apply_hamming_filter(data: &Array2<f64>, _sampling_freq: f64) -> KwaversResult<Array2<f64>> {
    let n = data.dim().1;
    let mut filtered = data.clone();

    for sensor_idx in 0..data.dim().0 {
        for i in 0..n {
            let window = 0.46f64.mul_add(-(2.0 * PI * i as f64 / (n - 1) as f64).cos(), 0.54);
            filtered[[sensor_idx, i]] *= window;
        }
    }

    Ok(filtered)
}

/// Hann window filter
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
fn apply_hann_filter(data: &Array2<f64>, _sampling_freq: f64) -> KwaversResult<Array2<f64>> {
    let n = data.dim().1;
    let mut filtered = data.clone();

    for sensor_idx in 0..data.dim().0 {
        for i in 0..n {
            let window = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos());
            filtered[[sensor_idx, i]] *= window;
        }
    }

    Ok(filtered)
}
