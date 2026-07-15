use super::*;
use crate::signal_processing::beamforming::covariance::CovariancePostProcess;
use approx::assert_abs_diff_eq;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::constants::numerical::TWO_PI;
use leto::Array3;

fn sensor_positions_m() -> Vec<[f64; 3]> {
    vec![
        [-0.015, 0.0, 0.0],
        [-0.005, 0.0, 0.0],
        [0.005, 0.0, 0.0],
        [0.015, 0.0, 0.0],
    ]
}

fn euclidean_distance_m(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn tof_s(sensor_pos: [f64; 3], source_pos: [f64; 3], sound_speed: f64) -> f64 {
    euclidean_distance_m(sensor_pos, source_pos) / sound_speed
}

fn synth_narrowband_sensor_data(
    sensor_positions: &[[f64; 3]],
    true_source: [f64; 3],
    sound_speed: f64,
    frequency_hz: f64,
    sampling_frequency_hz: f64,
    n_samples: usize,
    extra_delay_s: f64,
) -> Array3<f64> {
    let n_sensors = sensor_positions.len();
    let mut data = Array3::<f64>::zeros((n_sensors, 1, n_samples));

    let omega = TWO_PI * frequency_hz;

    for (i, &pos) in sensor_positions.iter().enumerate() {
        let tau = tof_s(pos, true_source, sound_speed) + extra_delay_s;
        for t in 0..n_samples {
            let time_s = (t as f64) / sampling_frequency_hz;
            data[[i, 0, t]] = (omega * (time_s - tau)).cos();
        }
    }

    data
}

#[test]
fn capon_spectrum_is_finite_for_simple_case() {
    let n_sensors = 2usize;
    let n_samples = 64usize;

    let mut x = Array3::<f64>::zeros((n_sensors, 1, n_samples));
    for t in 0..n_samples {
        x[[0, 0, t]] = 1.0;
        x[[1, 0, t]] = 1.0;
    }

    let positions = vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]];
    use crate::signal_processing::beamforming::covariance::CovarianceEstimator;
    use crate::signal_processing::beamforming::utils::steering::SteeringVectorMethod;
    let cfg = CaponSpectrumConfig {
        frequency_hz: MHZ_TO_HZ,
        sound_speed: SOUND_SPEED_WATER_SIM,
        diagonal_loading: 1e-3,
        covariance: CovarianceEstimator {
            forward_backward_averaging: false,
            num_snapshots: 1,
            post_process: CovariancePostProcess::None,
        },
        steering: SteeringVectorMethod::SphericalWave {
            source_position: [0.0, 0.0, 0.02],
        },
        sampling_frequency_hz: None,
        snapshot_selection: None,
        baseband_snapshot_step_samples: None,
    };

    let p = capon_spatial_spectrum_point(&x, &positions, [0.0, 0.0, 0.02], &cfg).expect("spectrum");
    assert!(p.is_finite());
    assert!(p > 0.0);
}

#[test]
fn complex_baseband_requires_sampling_frequency() {
    use crate::signal_processing::beamforming::covariance::CovarianceEstimator;
    use crate::signal_processing::beamforming::utils::steering::SteeringVectorMethod;

    let sensors = sensor_positions_m();
    let sound_speed = SOUND_SPEED_WATER_SIM;
    let sampling_frequency_hz = 2_000_000.0;
    let frequency_hz = 200_000.0;
    let n_samples = 256;

    let true_source = [0.0, 0.01, 0.02];
    let sensor_data = synth_narrowband_sensor_data(
        &sensors,
        true_source,
        sound_speed,
        frequency_hz,
        sampling_frequency_hz,
        n_samples,
        0.0,
    );

    let cfg = CaponSpectrumConfig {
        frequency_hz,
        sound_speed,
        diagonal_loading: 1e-3,
        covariance: CovarianceEstimator {
            forward_backward_averaging: false,
            num_snapshots: 1,
            post_process: CovariancePostProcess::None,
        },
        steering: SteeringVectorMethod::SphericalWave {
            source_position: true_source,
        },
        sampling_frequency_hz: None,
        snapshot_selection: None,
        baseband_snapshot_step_samples: None,
    };

    let err =
        capon_spatial_spectrum_point_complex_baseband(&sensor_data, &sensors, true_source, &cfg)
            .expect_err("missing sampling_frequency_hz must be rejected");
    assert!(err.to_string().contains("sampling_frequency_hz"));
}

#[test]
fn complex_baseband_rejects_invalid_snapshot_step() {
    use crate::signal_processing::beamforming::covariance::CovarianceEstimator;
    use crate::signal_processing::beamforming::utils::steering::SteeringVectorMethod;

    let sensors = sensor_positions_m();
    let sound_speed = SOUND_SPEED_WATER_SIM;
    let sampling_frequency_hz = 2_000_000.0;
    let frequency_hz = 200_000.0;
    let n_samples = 256;

    let true_source = [0.0, 0.01, 0.02];
    let sensor_data = synth_narrowband_sensor_data(
        &sensors,
        true_source,
        sound_speed,
        frequency_hz,
        sampling_frequency_hz,
        n_samples,
        0.0,
    );

    let cfg = CaponSpectrumConfig {
        frequency_hz,
        sound_speed,
        diagonal_loading: 1e-3,
        covariance: CovarianceEstimator {
            forward_backward_averaging: false,
            num_snapshots: 1,
            post_process: CovariancePostProcess::None,
        },
        steering: SteeringVectorMethod::SphericalWave {
            source_position: true_source,
        },
        sampling_frequency_hz: Some(sampling_frequency_hz),
        snapshot_selection: None,
        baseband_snapshot_step_samples: Some(0),
    };

    let err =
        capon_spatial_spectrum_point_complex_baseband(&sensor_data, &sensors, true_source, &cfg)
            .expect_err("snapshot step 0 must be rejected");
    assert!(err.to_string().contains("baseband_snapshot_step_samples"));
}

#[test]
fn complex_baseband_mvdr_is_invariant_to_global_time_shift() {
    use crate::signal_processing::beamforming::covariance::CovarianceEstimator;
    use crate::signal_processing::beamforming::utils::steering::SteeringVectorMethod;

    let sound_speed = SOUND_SPEED_WATER_SIM;
    let sampling_frequency_hz = 2_000_000.0;
    let frequency_hz = 200_000.0;
    let n_samples = 2048;

    let sensors = sensor_positions_m();
    let true_source = [0.0, 0.01, 0.02];

    let cfg = CaponSpectrumConfig {
        frequency_hz,
        sound_speed,
        diagonal_loading: 1e-3,
        covariance: CovarianceEstimator {
            forward_backward_averaging: false,
            num_snapshots: 1,
            post_process: CovariancePostProcess::None,
        },
        steering: SteeringVectorMethod::SphericalWave {
            source_position: true_source,
        },
        sampling_frequency_hz: Some(sampling_frequency_hz),
        snapshot_selection: None,
        baseband_snapshot_step_samples: Some(1),
    };

    let sensor_data_0 = synth_narrowband_sensor_data(
        &sensors,
        true_source,
        sound_speed,
        frequency_hz,
        sampling_frequency_hz,
        n_samples,
        0.0,
    );
    let sensor_data_shift = synth_narrowband_sensor_data(
        &sensors,
        true_source,
        sound_speed,
        frequency_hz,
        sampling_frequency_hz,
        n_samples,
        10.0 / frequency_hz,
    );

    let p0 =
        capon_spatial_spectrum_point_complex_baseband(&sensor_data_0, &sensors, true_source, &cfg)
            .expect("spectrum");
    let p1 = capon_spatial_spectrum_point_complex_baseband(
        &sensor_data_shift,
        &sensors,
        true_source,
        &cfg,
    )
    .expect("spectrum");

    assert!(p0.is_finite() && p0 > 0.0);
    assert!(p1.is_finite() && p1 > 0.0);
    assert_abs_diff_eq!(p0, p1, epsilon = 1e-6);
}
