use super::processor::DelayAndSumPAM;
use super::types::{ApodizationType, DelayAndSumConfig};
use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use approx::assert_relative_eq;
use ndarray::{Array1, Array2};

#[test]
fn test_pam_creation() {
    let sensors = vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]];
    let config = DelayAndSumConfig::default();
    let _pam = DelayAndSumPAM::new(sensors, config).unwrap();
}

#[test]
fn test_insufficient_sensors() {
    let sensors = vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]];
    let config = DelayAndSumConfig::default();
    assert!(DelayAndSumPAM::new(sensors, config).is_err());
}

#[test]
fn test_delay_computation() {
    let sensors = vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]];
    let config = DelayAndSumConfig::default();
    let pam = DelayAndSumPAM::new(sensors, config).unwrap();

    let source_pos = [0.0, 0.0, 0.0];
    let delays = pam.compute_delays(&source_pos).unwrap();

    assert_eq!(delays.len(), 3);
    assert_relative_eq!(delays[0], 0.0, epsilon = 1e-6);
}

#[test]
fn test_apodization_weights() {
    let sensors = vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]];
    let config = DelayAndSumConfig {
        apodization: ApodizationType::Uniform,
        ..Default::default()
    };
    let pam = DelayAndSumPAM::new(sensors, config).unwrap();

    let weights = pam.compute_apodization_weights();
    assert_eq!(weights.len(), 3);
    assert!(weights.iter().all(|&w| (w - 1.0).abs() < 1e-6));
}

#[test]
fn test_beamform_basic() {
    let sensors = vec![
        [0.0, 0.0, 0.0],
        [0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.01, 0.01, 0.0],
    ];
    let config = DelayAndSumConfig::default();
    let pam = DelayAndSumPAM::new(sensors, config).unwrap();

    let passive_data = Array2::<f64>::from_shape_fn((4, 1000), |(i, t)| {
        (2.0 * std::f64::consts::PI * t as f64 / 100.0 + i as f64).sin()
    });

    let grid_points = Array2::<f64>::from_shape_fn((5, 3), |(i, j)| match j {
        0 => (i as f64 - 2.0) * 0.005,
        1 => (i as f64 - 2.0) * 0.005,
        2 => 0.02,
        _ => 0.0,
    });

    let intensity_map = pam.beamform(&passive_data, &grid_points).unwrap();

    assert_eq!(intensity_map.len(), 5);
    assert!(intensity_map.iter().all(|&x| x >= 0.0));
}

#[test]
fn beamform_view_localizes_analytic_impulse_source() {
    let sound_speed = SOUND_SPEED_WATER_SIM;
    let sampling_frequency = 1.0e6;
    let sensors = vec![
        [-0.006, 0.0, 0.0],
        [-0.002, 0.0, 0.0],
        [0.002, 0.0, 0.0],
        [0.006, 0.0, 0.0],
    ];
    let source = [0.001, 0.0, 0.018];
    let distractor = [0.0, 0.0, 0.024];
    let config = DelayAndSumConfig {
        sound_speed,
        sampling_frequency,
        window_size: 64,
        apodization: ApodizationType::Uniform,
        ..Default::default()
    };
    let pam = DelayAndSumPAM::new(sensors.clone(), config).unwrap();
    let mut passive_data = Array2::<f64>::zeros((sensors.len(), 96));

    for (sensor_idx, sensor) in sensors.iter().enumerate() {
        let dx = source[0] - sensor[0];
        let dy = source[1] - sensor[1];
        let dz = source[2] - sensor[2];
        let sample = ((dx * dx + dy * dy + dz * dz).sqrt() / sound_speed * sampling_frequency)
            .round() as usize;
        passive_data[[sensor_idx, sample]] = 1.0;
    }

    let grid_points = Array2::from_shape_vec(
        (2, 3),
        vec![
            source[0],
            source[1],
            source[2],
            distractor[0],
            distractor[1],
            distractor[2],
        ],
    )
    .unwrap();
    let intensity = pam
        .beamform_view(passive_data.view(), grid_points.view())
        .unwrap();

    assert_eq!(intensity.len(), 2);
    assert!(intensity[0].is_finite());
    assert!(intensity[1].is_finite());
    assert!(
        intensity[0] > 4.0 * intensity[1],
        "true-source intensity {} did not dominate distractor {}",
        intensity[0],
        intensity[1]
    );
}

#[test]
fn beamform_view_uses_fractional_delay_interpolation() {
    let sensors = vec![[0.0, 0.0, 0.0]; 3];
    let config = DelayAndSumConfig {
        sound_speed: 1.0,
        sampling_frequency: 1.0,
        window_size: 1,
        apodization: ApodizationType::Uniform,
        ..Default::default()
    };
    let pam = DelayAndSumPAM::new(sensors, config).unwrap();
    let passive_data = Array2::from_shape_vec((3, 2), vec![0.0, 2.0, 0.0, 2.0, 0.0, 2.0]).unwrap();
    let grid_points = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.5]).unwrap();

    let intensity = pam
        .beamform_view(passive_data.view(), grid_points.view())
        .unwrap();

    assert_relative_eq!(intensity[0], 9.0, epsilon = 1e-12);
}

#[test]
fn beamform_view_rejects_invalid_boundary_shapes_and_values() {
    let sensors = vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]];
    let pam = DelayAndSumPAM::new(sensors, DelayAndSumConfig::default()).unwrap();
    let passive_data = Array2::<f64>::zeros((3, 16));
    let bad_grid_shape = Array2::<f64>::zeros((2, 2));
    assert!(pam
        .beamform_view(passive_data.view(), bad_grid_shape.view())
        .is_err());

    let mut nonfinite_data = passive_data.clone();
    nonfinite_data[[0, 0]] = f64::NAN;
    let grid_points = Array2::<f64>::zeros((1, 3));
    assert!(pam
        .beamform_view(nonfinite_data.view(), grid_points.view())
        .is_err());
}

#[test]
fn test_event_detection() {
    let sensors = vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]];
    let config = DelayAndSumConfig {
        detection_threshold: 2.0,
        ..Default::default()
    };
    let pam = DelayAndSumPAM::new(sensors, config).unwrap();

    let intensity_map = Array1::from_vec(vec![0.5, 0.8, 5.0, 1.0, 0.3]);
    let grid_points = Array2::from_shape_vec(
        (5, 3),
        vec![
            0.0, 0.0, 0.02, 0.01, 0.0, 0.02, 0.005, 0.005, 0.02, 0.0, 0.01, 0.02, -0.01, 0.0, 0.02,
        ],
    )
    .unwrap();

    let events = pam
        .detect_events(&intensity_map, &grid_points, 0.0)
        .unwrap();

    assert!(!events.is_empty());
    assert!(events[0].intensity > 2.0);
}

#[test]
fn test_event_detection_with_peak_frequency() {
    let sensors = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
    let config = DelayAndSumConfig {
        sampling_frequency: 10e6,
        window_size: 256,
        detection_threshold: 0.5,
        ..Default::default()
    };
    let pam = DelayAndSumPAM::new(sensors, config).unwrap();

    let freq = 1e6;
    let num_samples = 256;
    let mut passive_data = Array2::zeros((3, num_samples));
    for t in 0..num_samples {
        let time = t as f64 / pam.config.sampling_frequency;
        let sample = (2.0 * std::f64::consts::PI * freq * time).sin();
        for sensor in 0..3 {
            passive_data[[sensor, t]] = sample;
        }
    }

    let intensity_map = Array1::from_vec(vec![2.0]);
    let grid_points = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap();

    let events = pam
        .detect_events_with_data(&passive_data, &intensity_map, &grid_points, 0.0)
        .unwrap();

    assert!(!events.is_empty());
    let peak = events[0]
        .peak_frequency
        .expect("peak frequency should be available");
    let resolution = pam.config.sampling_frequency / num_samples as f64;
    assert!((peak - freq).abs() <= resolution);
}
