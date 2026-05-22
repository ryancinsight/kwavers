//! Mathematical invariance property tests for the narrowband pipeline.

use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use super::super::{
    capon::{capon_spatial_spectrum_point, CaponSpectrumConfig},
    snapshots::{SnapshotScenario, SnapshotSelection},
};
use super::helpers::{generate_plane_wave_data, generate_ula_positions};
use crate::analysis::signal_processing::beamforming::{
    covariance::{CovarianceEstimator, CovariancePostProcess},
    utils::steering::SteeringVectorMethod,
};
use ndarray::Array3;

#[test]
fn pipeline_is_invariant_to_global_time_shift() {
    // Mathematical property: Narrowband Capon spectrum should be invariant to
    // global time shifts (shifts all sensors equally), because it's based on
    // spatial covariance, not absolute time.

    let n_sensors = 4;
    let spacing_m = 0.0075;
    let n_samples = 128;
    let fs = 1_000_000.0;
    let f0 = 100_000.0;
    let c = SOUND_SPEED_WATER_SIM;
    let snr_db = 25.0;

    let data_original =
        generate_plane_wave_data(n_sensors, spacing_m, n_samples, fs, f0, 0.0, c, snr_db);

    // Create time-shifted version: shift by 10 samples
    let shift = 10;
    let mut data_shifted = Array3::<f64>::zeros((n_sensors, 1, n_samples));
    for sensor in 0..n_sensors {
        for t in shift..n_samples {
            data_shifted[(sensor, 0, t - shift)] = data_original[(sensor, 0, t)];
        }
    }

    let positions = generate_ula_positions(n_sensors, spacing_m);
    let candidate = [0.0, 0.0, 0.05];

    let scenario = SnapshotScenario {
        frequency_hz: f0,
        sampling_frequency_hz: fs,
        fractional_bandwidth: Some(0.05),
        prefer_robustness: true,
        prefer_time_resolution: false,
    };
    let selection = SnapshotSelection::Auto(scenario);

    let cfg = CaponSpectrumConfig {
        frequency_hz: f0,
        sound_speed: c,
        diagonal_loading: 1e-3,
        covariance: CovarianceEstimator {
            forward_backward_averaging: false,
            num_snapshots: 1,
            post_process: CovariancePostProcess::None,
        },
        steering: SteeringVectorMethod::SphericalWave {
            source_position: candidate,
        },
        sampling_frequency_hz: Some(fs),
        snapshot_selection: Some(selection),
        baseband_snapshot_step_samples: None,
    };

    let spectrum_original =
        capon_spatial_spectrum_point(&data_original, &positions, candidate, &cfg)
            .expect("spectrum original");
    let spectrum_shifted = capon_spatial_spectrum_point(&data_shifted, &positions, candidate, &cfg)
        .expect("spectrum shifted");

    // Spectra should be very similar (within 10% relative error)
    let rel_diff = (spectrum_original - spectrum_shifted).abs()
        / (spectrum_original + spectrum_shifted).max(1e-10);
    assert!(
        rel_diff < 0.1,
        "Spectra differ significantly under time shift: {:.2e} vs {:.2e} (rel diff: {:.2e})",
        spectrum_original,
        spectrum_shifted,
        rel_diff
    );
}
