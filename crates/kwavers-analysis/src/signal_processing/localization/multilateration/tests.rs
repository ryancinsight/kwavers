use super::solver::Multilateration;
use super::types::MultilaterationConfig;
use eunomia::assert_relative_eq;
use kwavers_core::constants::fundamental::{SOUND_SPEED_TISSUE, SOUND_SPEED_WATER_SIM};

#[test]
fn test_multilateration_creation() {
    let sensors = vec![
        [0.0, 0.0, 0.0],
        [0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, 0.0, 0.01],
    ];
    let config = MultilaterationConfig::default();
    let _multi = Multilateration::new(sensors, config).unwrap();
}

#[test]
fn test_insufficient_sensors() {
    let sensors = vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]];
    let config = MultilaterationConfig::default();
    assert!(Multilateration::new(sensors, config).is_err());
}

#[test]
fn test_set_uncertainties() {
    let sensors = vec![
        [0.0, 0.0, 0.0],
        [0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, 0.0, 0.01],
    ];
    let config = MultilaterationConfig::default();
    let mut multi = Multilateration::new(sensors, config).unwrap();

    let uncertainties = vec![1e-9, 1e-9, 1e-9, 1e-9];
    multi.set_sensor_uncertainties(uncertainties).unwrap();

    let bad_uncertainties = vec![1e-9, 1e-9];
    assert!(multi.set_sensor_uncertainties(bad_uncertainties).is_err());
}

#[test]
fn test_localize_symmetric_array() {
    let c = SOUND_SPEED_WATER_SIM;
    let sensors = vec![
        [0.01, 0.0, 0.0],
        [-0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, -0.01, 0.0],
        [0.0, 0.0, 0.01],
    ];

    let config = MultilaterationConfig {
        sound_speed: c,
        initial_guess: Some([0.0, 0.0, 0.0]),
        ..Default::default()
    };
    let multi = Multilateration::new(sensors.clone(), config).unwrap();

    let source_pos = [0.0, 0.0, 0.0];
    let arrival_times: Vec<f64> = sensors
        .iter()
        .map(|s| {
            let dx = source_pos[0] - s[0];
            let dy = source_pos[1] - s[1];
            let dz = source_pos[2] - s[2];
            (dx * dx + dy * dy + dz * dz).sqrt() / c
        })
        .collect();

    let result = multi.localize(&arrival_times).unwrap();

    assert!(result.converged);
    assert_relative_eq!(result.position[0], 0.0, epsilon = 1e-4);
    assert_relative_eq!(result.position[1], 0.0, epsilon = 1e-4);
    assert_relative_eq!(result.position[2], 0.0, epsilon = 1e-4);
    assert!(result.residual < 1e-5);
}

#[test]
fn test_localize_off_axis_overdetermined() {
    let c = SOUND_SPEED_TISSUE;
    let sensors = vec![
        [0.0, 0.0, 0.0],
        [0.02, 0.0, 0.0],
        [0.0, 0.02, 0.0],
        [0.0, 0.0, 0.02],
        [0.01, 0.01, 0.0],
        [0.01, 0.0, 0.01],
    ];

    let config = MultilaterationConfig {
        sound_speed: c,
        ..Default::default()
    };
    let multi = Multilateration::new(sensors.clone(), config).unwrap();

    let source_pos = [0.005, 0.005, 0.005];
    let arrival_times: Vec<f64> = sensors
        .iter()
        .map(|s| {
            let dx = source_pos[0] - s[0];
            let dy = source_pos[1] - s[1];
            let dz = source_pos[2] - s[2];
            (dx * dx + dy * dy + dz * dz).sqrt() / c
        })
        .collect();

    let result = multi.localize(&arrival_times).unwrap();

    assert!(result.converged);
    assert_relative_eq!(result.position[0], source_pos[0], epsilon = 1e-4);
    assert_relative_eq!(result.position[1], source_pos[1], epsilon = 1e-4);
    assert_relative_eq!(result.position[2], source_pos[2], epsilon = 1e-4);
}

#[test]
fn test_weighted_least_squares() {
    let c = SOUND_SPEED_WATER_SIM;
    let sensors = vec![
        [0.0, 0.0, 0.0],
        [0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, 0.0, 0.01],
        [-0.01, 0.0, 0.0],
    ];

    let config = MultilaterationConfig {
        sound_speed: c,
        use_weighted_ls: true,
        ..Default::default()
    };
    let mut multi = Multilateration::new(sensors.clone(), config).unwrap();

    let uncertainties = vec![1e-10, 1e-10, 5e-10, 5e-10, 5e-10];
    multi.set_sensor_uncertainties(uncertainties).unwrap();

    let source_pos = [0.002, 0.002, 0.002];
    let arrival_times: Vec<f64> = sensors
        .iter()
        .map(|s| {
            let dx = source_pos[0] - s[0];
            let dy = source_pos[1] - s[1];
            let dz = source_pos[2] - s[2];
            (dx * dx + dy * dy + dz * dz).sqrt() / c
        })
        .collect();

    let result = multi.localize(&arrival_times).unwrap();

    assert!(result.converged);
    assert_relative_eq!(result.position[0], source_pos[0], epsilon = 1e-4);
    assert_relative_eq!(result.position[1], source_pos[1], epsilon = 1e-4);
    assert_relative_eq!(result.position[2], source_pos[2], epsilon = 1e-4);
}

#[test]
fn test_gdop_calculation() {
    let c = SOUND_SPEED_WATER_SIM;
    let sensors = vec![
        [0.01, 0.0, 0.0],
        [-0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, -0.01, 0.0],
        [0.0, 0.0, 0.01],
    ];

    let config = MultilaterationConfig {
        sound_speed: c,
        ..Default::default()
    };
    let multi = Multilateration::new(sensors, config).unwrap();

    let source_pos = [0.0, 0.0, 0.0];
    let gdop = multi.calculate_gdop(&source_pos).unwrap();

    assert!(gdop > 0.0);
    assert!(gdop < 10.0);
}

#[test]
fn test_noisy_measurements() {
    let c = SOUND_SPEED_TISSUE;
    let sensors = vec![
        [0.0, 0.0, 0.0],
        [0.02, 0.0, 0.0],
        [0.0, 0.02, 0.0],
        [0.0, 0.0, 0.02],
        [0.01, 0.01, 0.01],
    ];

    let config = MultilaterationConfig {
        sound_speed: c,
        max_iterations: 100,
        ..Default::default()
    };
    let multi = Multilateration::new(sensors.clone(), config).unwrap();

    let source_pos = [0.008, 0.008, 0.008];
    let noise = [0.0, 1e-9, -1e-9, 0.5e-9, -0.5e-9];
    let arrival_times: Vec<f64> = sensors
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let dx = source_pos[0] - s[0];
            let dy = source_pos[1] - s[1];
            let dz = source_pos[2] - s[2];
            (dx * dx + dy * dy + dz * dz).sqrt() / c + noise[i]
        })
        .collect();

    let result = multi.localize(&arrival_times).unwrap();

    assert!(result.converged);
    assert_relative_eq!(result.position[0], source_pos[0], epsilon = 1e-3);
    assert_relative_eq!(result.position[1], source_pos[1], epsilon = 1e-3);
    assert_relative_eq!(result.position[2], source_pos[2], epsilon = 1e-3);
}
