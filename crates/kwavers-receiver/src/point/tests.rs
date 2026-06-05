use super::*;
use approx::assert_abs_diff_eq;
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::constants::numerical::TWO_PI;
use ndarray::Array3;

fn create_test_grid() -> Grid {
    Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap()
}

#[test]
fn test_point_sensor_creation() {
    let grid = create_test_grid();
    let locations = vec![[0.016, 0.016, 0.016]];

    let config = PointSensorConfig::new(locations);
    let sensor = PointSensor::new(config, &grid).unwrap();

    assert_eq!(sensor.n_sensors(), 1);
    assert_eq!(sensor.n_timesteps(), 0);
}

#[test]
fn test_point_sensor_validation() {
    let grid = create_test_grid();

    let locations = vec![[0.100, 0.016, 0.016]];
    let config = PointSensorConfig::new(locations);
    assert!(PointSensor::new(config, &grid).is_err());

    let locations = vec![[f64::NAN, 0.016, 0.016]];
    let config = PointSensorConfig::new(locations);
    assert!(PointSensor::new(config, &grid).is_err());

    let config = PointSensorConfig::new(vec![]);
    assert!(PointSensor::new(config, &grid).is_err());
}

#[test]
fn test_trilinear_interpolation_at_grid_point() {
    let grid = create_test_grid();
    let field = Array3::<f64>::from_shape_fn((32, 32, 32), |(i, j, k)| (i + j + k) as f64);

    let locations = vec![[0.016, 0.016, 0.016]];
    let config = PointSensorConfig::new(locations);
    let mut sensor = PointSensor::new(config, &grid).unwrap();

    sensor.record(field.view(), &grid, 0);

    let history = sensor.time_history(0).unwrap();
    let expected = field[[16, 16, 16]];
    assert_abs_diff_eq!(history[0], expected, epsilon = 1e-12);
}

#[test]
fn test_trilinear_interpolation_midpoint() {
    let grid = create_test_grid();

    let field = Array3::<f64>::from_shape_fn((32, 32, 32), |(i, j, k)| {
        (i as f64) + 10.0 * (j as f64) + 100.0 * (k as f64)
    });

    let locations = vec![[0.0105, 0.0105, 0.0105]];
    let config = PointSensorConfig::new(locations);
    let mut sensor = PointSensor::new(config, &grid).unwrap();

    sensor.record(field.view(), &grid, 0);

    let corners = [
        field[[10, 10, 10]],
        field[[11, 10, 10]],
        field[[10, 11, 10]],
        field[[11, 11, 10]],
        field[[10, 10, 11]],
        field[[11, 10, 11]],
        field[[10, 11, 11]],
        field[[11, 11, 11]],
    ];
    let expected: f64 = corners.iter().sum::<f64>() / 8.0;

    let history = sensor.time_history(0).unwrap();
    assert_abs_diff_eq!(history[0], expected, epsilon = 1e-9);
}

#[test]
fn test_multiple_sensors() {
    let grid = create_test_grid();
    let locations = vec![
        [0.016, 0.016, 0.016],
        [0.020, 0.016, 0.016],
        [0.016, 0.020, 0.016],
    ];

    let config = PointSensorConfig::new(locations);
    let mut sensor = PointSensor::new(config, &grid).unwrap();

    assert_eq!(sensor.n_sensors(), 3);

    for t in 0..10 {
        let field = Array3::<f64>::from_elem((32, 32, 32), (t as f64) * 10.0);
        sensor.record(field.view(), &grid, t);
    }

    assert_eq!(sensor.n_timesteps(), 10);

    for i in 0..3 {
        let history = sensor.time_history(i).unwrap();
        assert_eq!(history.len(), 10);
    }
}

#[test]
fn test_time_history_recording() {
    let grid = create_test_grid();
    let locations = vec![[0.016, 0.016, 0.016]];
    let config = PointSensorConfig::new(locations);
    let mut sensor = PointSensor::new(config, &grid).unwrap();

    let omega = TWO_PI * MHZ_TO_HZ;
    let dt = 1e-7;

    for t in 0..100 {
        let time = (t as f64) * dt;
        let pressure = (omega * time).sin();
        let field = Array3::<f64>::from_elem((32, 32, 32), pressure);
        sensor.record(field.view(), &grid, t);
    }

    let history = sensor.time_history(0).unwrap();
    assert_eq!(history.len(), 100);

    for t in 0..5 {
        let time = (t as f64) * dt;
        let expected = (omega * time).sin();
        assert_abs_diff_eq!(history[t], expected, epsilon = 1e-12);
    }
}

#[test]
fn test_max_and_rms_pressure() {
    let grid = create_test_grid();
    let locations = vec![[0.016, 0.016, 0.016]];
    let config = PointSensorConfig::new(locations);
    let mut sensor = PointSensor::new(config, &grid).unwrap();

    let values = vec![1.0, -3.0, 2.0, -1.0, 0.0];
    for &val in &values {
        let field = Array3::<f64>::from_elem((32, 32, 32), val);
        sensor.record(field.view(), &grid, 0);
    }

    let max_p = sensor.max_pressure(0).unwrap();
    assert_abs_diff_eq!(max_p, 3.0, epsilon = 1e-12);

    let rms_p = sensor.rms_pressure(0).unwrap();
    let expected_rms = (15.0_f64 / 5.0).sqrt();
    assert_abs_diff_eq!(rms_p, expected_rms, epsilon = 1e-12);
}

#[test]
fn test_clear_history() {
    let grid = create_test_grid();
    let locations = vec![[0.016, 0.016, 0.016]];
    let config = PointSensorConfig::new(locations);
    let mut sensor = PointSensor::new(config, &grid).unwrap();

    for t in 0..10 {
        let field = Array3::<f64>::from_elem((32, 32, 32), t as f64);
        sensor.record(field.view(), &grid, t);
    }

    assert_eq!(sensor.n_timesteps(), 10);

    sensor.clear();

    assert_eq!(sensor.n_timesteps(), 0);
    assert_eq!(sensor.time_history(0).unwrap().len(), 0);
}

#[test]
fn test_csv_export() {
    let grid = create_test_grid();
    let locations = vec![[0.016, 0.016, 0.016], [0.020, 0.016, 0.016]];
    let config = PointSensorConfig::new(locations);
    let mut sensor = PointSensor::new(config, &grid).unwrap();

    for t in 0..3 {
        let field = Array3::<f64>::from_elem((32, 32, 32), (t as f64) * 10.0);
        sensor.record(field.view(), &grid, t);
    }

    let dt = 1e-7;
    let csv = sensor.to_csv(dt);

    assert!(csv.contains("time,sensor_0,sensor_1"));
    assert!(csv.contains("0.000000e0,0.000000e0,0.000000e0"));
}

#[test]
fn test_all_time_histories() {
    let grid = create_test_grid();
    let locations = vec![[0.016, 0.016, 0.016], [0.020, 0.016, 0.016]];
    let config = PointSensorConfig::new(locations);
    let mut sensor = PointSensor::new(config, &grid).unwrap();

    for t in 0..5 {
        let field = Array3::<f64>::from_elem((32, 32, 32), (t as f64) * 10.0);
        sensor.record(field.view(), &grid, t);
    }

    let histories = sensor.all_time_histories();
    assert_eq!(histories.shape(), &[2, 5]);

    for t in 0..5 {
        let expected = (t as f64) * 10.0;
        assert_abs_diff_eq!(histories[[0, t]], expected, epsilon = 1e-12);
        assert_abs_diff_eq!(histories[[1, t]], expected, epsilon = 1e-12);
    }
}
