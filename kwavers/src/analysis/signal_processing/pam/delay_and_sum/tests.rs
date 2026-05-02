use super::processor::DelayAndSumPAM;
use super::types::{ApodizationType, DelayAndSumConfig};
use approx::assert_relative_eq;
use ndarray::{Array1, Array2};

#[test]
fn test_pam_creation() {
    let sensors = vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]];
    let config = DelayAndSumConfig::default();
    let pam = DelayAndSumPAM::new(sensors, config);
    assert!(pam.is_ok());
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
        apodization: ApodizationType::None,
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
            0.0, 0.0, 0.02, 0.01, 0.0, 0.02, 0.005, 0.005, 0.02, 0.0, 0.01, 0.02, -0.01, 0.0,
            0.02,
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
