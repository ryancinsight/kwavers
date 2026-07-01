//! End-to-end pipeline and diagonal-loading integration tests.

use super::super::{
    capon::{capon_spatial_spectrum_point, CaponSpectrumConfig},
    snapshots::{extract_narrowband_snapshots, SnapshotScenario, SnapshotSelection},
};
use super::helpers::{generate_plane_wave_data, generate_ula_positions, PlaneWaveDataSpec};
use crate::signal_processing::beamforming::{
    covariance::{CovarianceEstimator, CovariancePostProcess},
    utils::steering::SteeringVectorMethod,
};
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::constants::numerical::MHZ_TO_HZ;

#[test]
fn end_to_end_pipeline_produces_finite_spectrum() {
    // Setup: 4-element ULA, plane wave from broadside
    let n_sensors = 4;
    let spacing_m = 0.0075; // λ/2 at 100 kHz in water
    let n_samples = 128;
    let fs = MHZ_TO_HZ; // 1 MHz sampling
    let f0 = 100_000.0; // 100 kHz signal
    let c = SOUND_SPEED_WATER_SIM; // sound speed
    let snr_db = 20.0;

    let data = generate_plane_wave_data(PlaneWaveDataSpec {
        n_sensors,
        sensor_spacing_m: spacing_m,
        n_samples,
        sampling_frequency_hz: fs,
        signal_frequency_hz: f0,
        angle_deg: 0.0,
        sound_speed_m_per_s: c,
        snr_db,
    });
    let positions = generate_ula_positions(n_sensors, spacing_m);

    // Pipeline step 1: Extract snapshots
    let scenario = SnapshotScenario {
        frequency_hz: f0,
        sampling_frequency_hz: fs,
        fractional_bandwidth: Some(0.05),
        prefer_robustness: true,
        prefer_time_resolution: false,
    };
    let selection = SnapshotSelection::Auto(scenario);
    let snapshots = extract_narrowband_snapshots(&data, &selection).expect("snapshot extraction");

    assert!(
        snapshots.nrows() == n_sensors,
        "snapshot dimensions mismatch"
    );
    assert!(snapshots.ncols() > 0, "no snapshots extracted");

    // Pipeline step 2: Compute Capon spectrum at broadside
    let cfg = CaponSpectrumConfig {
        frequency_hz: f0,
        sound_speed: c,
        diagonal_loading: 1e-3,
        covariance: CovarianceEstimator {
            forward_backward_averaging: false,
            num_snapshots: snapshots.ncols(),
            post_process: CovariancePostProcess::None,
        },
        steering: SteeringVectorMethod::SphericalWave {
            source_position: [0.0, 0.0, 0.05], // 5 cm in front
        },
        sampling_frequency_hz: Some(fs),
        snapshot_selection: Some(selection),
        baseband_snapshot_step_samples: None,
    };

    let spectrum = capon_spatial_spectrum_point(&data, &positions, [0.0, 0.0, 0.05], &cfg)
        .expect("capon spectrum");

    // Validation: spectrum must be finite and positive
    assert!(spectrum.is_finite(), "spectrum is not finite");
    assert!(spectrum > 0.0, "spectrum must be positive");
}

#[test]
fn capon_spectrum_varies_across_candidate_grid() {
    // Test that Capon spectrum produces different values for different candidate points
    let n_sensors = 8;
    let spacing_m = 0.0075; // λ/2 at 100 kHz
    let n_samples = 256;
    let fs = MHZ_TO_HZ;
    let f0 = 100_000.0;
    let c = SOUND_SPEED_WATER_SIM;
    let snr_db = 30.0; // High SNR

    let data = generate_plane_wave_data(PlaneWaveDataSpec {
        n_sensors,
        sensor_spacing_m: spacing_m,
        n_samples,
        sampling_frequency_hz: fs,
        signal_frequency_hz: f0,
        angle_deg: 0.0,
        sound_speed_m_per_s: c,
        snr_db,
    });
    let positions = generate_ula_positions(n_sensors, spacing_m);

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
            forward_backward_averaging: true,
            num_snapshots: 1,
            post_process: CovariancePostProcess::None,
        },
        steering: SteeringVectorMethod::SphericalWave {
            source_position: [0.0, 0.0, 0.05],
        },
        sampling_frequency_hz: Some(fs),
        snapshot_selection: Some(selection),
        baseband_snapshot_step_samples: None,
    };

    // Test multiple candidate points
    let candidates = vec![
        [0.0, 0.0, 0.05],   // Center (should be highest for broadside source)
        [0.02, 0.0, 0.05],  // Off-axis
        [-0.02, 0.0, 0.05], // Off-axis opposite
    ];

    let mut spectra = Vec::new();
    for &candidate in &candidates {
        let mut cfg_scan = cfg.clone();
        cfg_scan.steering = SteeringVectorMethod::SphericalWave {
            source_position: candidate,
        };

        let spectrum = capon_spatial_spectrum_point(&data, &positions, candidate, &cfg_scan)
            .expect("capon spectrum");
        spectra.push(spectrum);

        assert!(
            spectrum.is_finite(),
            "Spectrum not finite for candidate {:?}",
            candidate
        );
        assert!(
            spectrum > 0.0,
            "Spectrum not positive for candidate {:?}",
            candidate
        );
    }

    // Verify spectra are not all identical (shows discrimination)
    let max_spectrum = spectra.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_spectrum = spectra.iter().cloned().fold(f64::INFINITY, f64::min);
    let dynamic_range = max_spectrum / min_spectrum.max(1e-10);

    assert!(
        dynamic_range > 1.1,
        "Capon spectrum shows insufficient spatial discrimination: dynamic range {:.2}",
        dynamic_range
    );
}

#[test]
fn diagonal_loading_prevents_covariance_singularity() {
    // Test that diagonal loading stabilizes Capon spectrum computation
    let n_sensors = 3;
    let spacing_m = 0.0075;
    let n_samples = 32; // Few samples → poorly conditioned covariance
    let fs = MHZ_TO_HZ;
    let f0 = 100_000.0;
    let c = SOUND_SPEED_WATER_SIM;
    let snr_db = 10.0; // Low SNR

    let data = generate_plane_wave_data(PlaneWaveDataSpec {
        n_sensors,
        sensor_spacing_m: spacing_m,
        n_samples,
        sampling_frequency_hz: fs,
        signal_frequency_hz: f0,
        angle_deg: 0.0,
        sound_speed_m_per_s: c,
        snr_db,
    });
    let positions = generate_ula_positions(n_sensors, spacing_m);
    let candidate = [0.0, 0.0, 0.05];

    let scenario = SnapshotScenario {
        frequency_hz: f0,
        sampling_frequency_hz: fs,
        fractional_bandwidth: Some(0.05),
        prefer_robustness: true,
        prefer_time_resolution: false,
    };

    // Test with no loading
    let cfg_no_loading = CaponSpectrumConfig {
        frequency_hz: f0,
        sound_speed: c,
        diagonal_loading: 0.0,
        covariance: CovarianceEstimator {
            forward_backward_averaging: false,
            num_snapshots: 1,
            post_process: CovariancePostProcess::None,
        },
        steering: SteeringVectorMethod::SphericalWave {
            source_position: candidate,
        },
        sampling_frequency_hz: Some(fs),
        snapshot_selection: Some(SnapshotSelection::Auto(scenario)),
        baseband_snapshot_step_samples: None,
    };

    // Test with modest loading
    let cfg_with_loading = CaponSpectrumConfig {
        diagonal_loading: 1e-2,
        ..cfg_no_loading.clone()
    };

    let spectrum_no_loading =
        capon_spatial_spectrum_point(&data, &positions, candidate, &cfg_no_loading)
            .expect("spectrum without loading");
    let spectrum_with_loading =
        capon_spatial_spectrum_point(&data, &positions, candidate, &cfg_with_loading)
            .expect("spectrum with loading");

    assert!(spectrum_no_loading.is_finite() && spectrum_no_loading > 0.0);
    assert!(spectrum_with_loading.is_finite() && spectrum_with_loading > 0.0);
    assert!(
        spectrum_with_loading > 0.0,
        "Diagonal loading should produce positive spectrum"
    );
}
