use crate::analysis::signal_processing::localization::model_order::ModelOrderCriterion;
use crate::analysis::signal_processing::localization::{LocalizationConfig, LocalizationProcessor};
use ndarray::Array2;
use num_complex::Complex;

use super::{MUSICConfig, MUSICProcessor};

#[test]
fn test_music_processor_creation() {
    let config = MUSICConfig::default();
    let result = MUSICProcessor::new(&config);
    assert!(result.is_ok());
}

#[test]
fn test_music_invalid_num_sources_zero() {
    let mut config = MUSICConfig::default();
    config.num_sources = Some(0);
    let result = MUSICProcessor::new(&config);
    assert!(result.is_err());
}

#[test]
fn test_music_invalid_num_sources_too_many() {
    let mut config = MUSICConfig::default();
    config.num_sources = Some(10); // More than sensors
    let result = MUSICProcessor::new(&config);
    assert!(result.is_err());
}

#[test]
fn test_music_config_builder() {
    let config = MUSICConfig::default()
        .with_grid_resolution(100)
        .with_min_separation(0.05)
        .with_criterion(ModelOrderCriterion::AIC);

    assert_eq!(config.grid_resolution, 100);
    assert_eq!(config.min_source_separation, 0.05);
    assert_eq!(config.model_order_criterion, ModelOrderCriterion::AIC);
}

#[test]
fn test_covariance_estimation() {
    let config = MUSICConfig::default();
    let processor = MUSICProcessor::new(&config).unwrap();

    let snapshots = Array2::from_shape_fn((2, 10), |(i, j)| Complex::new((i + j) as f64, 0.0));

    let cov = processor.estimate_covariance(&snapshots).unwrap();

    assert_eq!(cov.dim(), (2, 2));

    // Check Hermitian property: R[i,j] = conj(R[j,i])
    for i in 0..2 {
        for j in 0..2 {
            assert!((cov[[i, j]] - cov[[j, i]].conj()).norm() < 1e-10);
        }
    }
}

#[test]
fn test_steering_vector() {
    let source_pos = [0.0, 0.0, 0.0];
    let sensor_positions = vec![[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]];
    let frequency = 1000.0;
    let speed_of_sound = 1500.0;

    let steering =
        MUSICProcessor::steering_vector(source_pos, &sensor_positions, frequency, speed_of_sound);

    assert_eq!(steering.len(), 3);

    // All sensors at source location should have unit magnitude
    for val in steering.iter() {
        assert!((val.norm() - 1.0).abs() < 1e-10);
    }
}

#[test]
fn test_music_run_single_source() {
    let mut config = MUSICConfig::default();
    config.num_sources = Some(1);
    config.grid_resolution = 10;
    config.num_snapshots = 50;

    let processor = MUSICProcessor::new(&config).unwrap();

    let num_sensors = config.config.sensor_positions.len();
    let snapshots = Array2::from_shape_fn((num_sensors, 50), |(i, j)| {
        Complex::new((i + j) as f64 / 10.0, (i * j) as f64 / 20.0)
    });

    let result = processor.run(&snapshots);
    assert!(result.is_ok());

    let music_result = result.unwrap();
    assert_eq!(music_result.num_sources, 1);
    assert!(music_result.sources.len() <= 1);
}

#[test]
fn test_music_automatic_source_detection() {
    let mut config = MUSICConfig::default();
    config.num_sources = None;
    config.num_snapshots = 100;
    config.model_order_criterion = ModelOrderCriterion::MDL;

    let processor = MUSICProcessor::new(&config).unwrap();

    let num_sensors = 4;
    let snapshots = Array2::from_shape_fn((num_sensors, 100), |(i, j)| {
        if i < 2 {
            Complex::new(10.0 * (i + j) as f64, 0.0)
        } else {
            Complex::new((i + j) as f64 / 10.0, 0.0)
        }
    });

    let result = processor.run(&snapshots);
    assert!(result.is_ok());
}

#[test]
fn music_trait_localize_uses_time_delay_physics() {
    let c = 1500.0;
    let sensor_positions = vec![
        [0.01, 0.0, 0.0],
        [-0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, 0.0, 0.01],
    ];
    let source = [0.002, 0.003, 0.004];
    let arrival_times: Vec<f64> = sensor_positions
        .iter()
        .map(|sensor| {
            let dx = source[0] - sensor[0];
            let dy = source[1] - sensor[1];
            let dz = source[2] - sensor[2];
            let squared_distance: f64 = dx * dx + dy * dy + dz * dz;
            squared_distance.sqrt() / c
        })
        .collect();

    let localization_config = LocalizationConfig::new(sensor_positions.clone(), 40.0e6, c);
    let music_config = MUSICConfig::new(localization_config, Some(1));
    let processor = MUSICProcessor::new(&music_config).unwrap();

    let result = processor
        .localize(&arrival_times, &sensor_positions)
        .unwrap();

    for axis in 0..3 {
        assert!((result.position[axis] - source[axis]).abs() < 1e-4);
    }
    assert!(result.confidence > 0.0);
    assert!(result.uncertainty < 1e-4);
}

#[test]
fn music_trait_localize_rejects_delay_sensor_count_mismatch() {
    let processor = MUSICProcessor::new(&MUSICConfig::default()).unwrap();
    let error = processor
        .localize(
            &[0.0],
            &[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]],
        )
        .unwrap_err();

    assert!(format!("{error}").contains("one arrival time per sensor"));
}
