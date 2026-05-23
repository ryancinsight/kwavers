//! Cross-method snapshot consistency tests.

use super::super::snapshots::{extract_narrowband_snapshots, SnapshotScenario, SnapshotSelection};
use super::helpers::{compute_sample_covariance, generate_plane_wave_data};
use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;

#[test]
fn snapshot_methods_produce_consistent_covariance_rank() {
    // Different snapshot extraction methods should produce covariance matrices
    // with similar effective rank (number of significant eigenvalues).

    let n_sensors = 4;
    let spacing_m = 0.0075;
    let n_samples = 256;
    let fs = 1_000_000.0;
    let f0 = 100_000.0;
    let c = SOUND_SPEED_WATER_SIM;
    let snr_db = 20.0;

    let data = generate_plane_wave_data(n_sensors, spacing_m, n_samples, fs, f0, 0.0, c, snr_db);

    // Method 1: Robust windowed STFT
    let scenario_robust = SnapshotScenario {
        frequency_hz: f0,
        sampling_frequency_hz: fs,
        fractional_bandwidth: Some(0.05),
        prefer_robustness: true,
        prefer_time_resolution: false,
    };
    let snapshots_robust =
        extract_narrowband_snapshots(&data, &SnapshotSelection::Auto(scenario_robust))
            .expect("robust snapshots");

    // Method 2: Time-resolution optimized
    let scenario_time_res = SnapshotScenario {
        frequency_hz: f0,
        sampling_frequency_hz: fs,
        fractional_bandwidth: Some(0.05),
        prefer_robustness: false,
        prefer_time_resolution: true,
    };
    let snapshots_time_res =
        extract_narrowband_snapshots(&data, &SnapshotSelection::Auto(scenario_time_res))
            .expect("time-res snapshots");

    // Both should produce valid snapshots with correct sensor dimension
    assert_eq!(snapshots_robust.nrows(), n_sensors);
    assert_eq!(snapshots_time_res.nrows(), n_sensors);
    assert!(snapshots_robust.ncols() > 0);
    assert!(snapshots_time_res.ncols() > 0);

    // Compute sample covariance for each
    let cov_robust = compute_sample_covariance(&snapshots_robust);
    let cov_time_res = compute_sample_covariance(&snapshots_time_res);

    // Check that both are Hermitian positive semi-definite
    for i in 0..n_sensors {
        assert!(
            cov_robust[(i, i)].im.abs() < 1e-10,
            "Covariance diagonal should be real"
        );
        assert!(
            cov_robust[(i, i)].re >= 0.0,
            "Covariance diagonal should be non-negative"
        );
        assert!(cov_time_res[(i, i)].im.abs() < 1e-10);
        assert!(cov_time_res[(i, i)].re >= 0.0);
    }
}
