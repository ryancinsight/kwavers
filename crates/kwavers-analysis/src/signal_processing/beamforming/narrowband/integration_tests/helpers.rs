//! Shared helper functions for narrowband integration tests.

use kwavers_core::constants::numerical::TWO_PI;
use ndarray::Array3;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Generate synthetic array data with a plane wave from a known direction.
///
/// # Mathematical Model
///
/// For a plane wave at angle θ relative to the array axis:
/// ```text
/// x_m(t) = cos(2πf₀t + k·d_m·sin(θ))
/// ```
/// where:
/// - f₀ = carrier frequency
/// - k = 2πf₀/c (wavenumber)
/// - d_m = position of sensor m
/// - θ = angle of arrival
pub(super) fn generate_plane_wave_data(
    n_sensors: usize,
    sensor_spacing_m: f64,
    n_samples: usize,
    sampling_frequency_hz: f64,
    signal_frequency_hz: f64,
    angle_deg: f64,
    sound_speed_m_per_s: f64,
    snr_db: f64,
) -> Array3<f64> {
    let mut data = Array3::<f64>::zeros((n_sensors, 1, n_samples));

    let angle_rad = angle_deg * PI / 180.0;
    let k = TWO_PI * signal_frequency_hz / sound_speed_m_per_s;

    // Signal power (assuming unit amplitude)
    let signal_power = 0.5; // RMS power of cos wave with amplitude 1
    let noise_power = signal_power / 10.0_f64.powf(snr_db / 10.0);
    let noise_std = noise_power.sqrt();

    for sensor_idx in 0..n_sensors {
        let sensor_position_m = sensor_idx as f64 * sensor_spacing_m;
        let phase_shift = k * sensor_position_m * angle_rad.sin();

        for sample_idx in 0..n_samples {
            let t = sample_idx as f64 / sampling_frequency_hz;
            let signal = (TWO_PI * signal_frequency_hz * t + phase_shift).cos();

            // Add white Gaussian noise (simplified: use deterministic pseudo-noise)
            let noise = noise_std
                * ((sample_idx as f64 * 17.0 + sensor_idx as f64 * 23.0).sin()
                    + (sample_idx as f64 * 13.0 + sensor_idx as f64 * 29.0).cos())
                / 2.0_f64.sqrt();

            data[(sensor_idx, 0, sample_idx)] = signal + noise;
        }
    }

    data
}

/// Generate sensor positions for a uniform linear array.
pub(super) fn generate_ula_positions(n_sensors: usize, spacing_m: f64) -> Vec<[f64; 3]> {
    (0..n_sensors)
        .map(|i| [i as f64 * spacing_m, 0.0, 0.0])
        .collect()
}

/// Compute sample covariance matrix from complex snapshots.
pub(super) fn compute_sample_covariance(
    snapshots: &ndarray::Array2<Complex64>,
) -> ndarray::Array2<Complex64> {
    let n_sensors = snapshots.nrows();
    let n_snapshots = snapshots.ncols();
    let mut cov = ndarray::Array2::<Complex64>::zeros((n_sensors, n_sensors));

    for k in 0..n_snapshots {
        let snapshot = snapshots.column(k);
        for i in 0..n_sensors {
            for j in 0..n_sensors {
                cov[(i, j)] += snapshot[i] * snapshot[j].conj();
            }
        }
    }

    // Normalize
    for elem in cov.iter_mut() {
        *elem /= n_snapshots as f64;
    }

    cov
}
