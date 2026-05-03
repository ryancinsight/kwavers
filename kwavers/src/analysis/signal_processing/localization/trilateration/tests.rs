use super::solver::Trilateration;
use super::types::TrilaterationConfig;
use approx::assert_relative_eq;

#[test]
fn test_trilateration_creation() {
    let sensors = vec![
        [0.0, 0.0, 0.0],
        [0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, 0.0, 0.01],
    ];
    let config = TrilaterationConfig::default();
    let trilat = Trilateration::new(sensors, config);
    assert!(trilat.is_ok());
}

#[test]
fn test_insufficient_sensors() {
    let sensors = vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]];
    let config = TrilaterationConfig::default();
    assert!(Trilateration::new(sensors, config).is_err());
}

#[test]
fn test_localize_source_at_origin() {
    let c = 1500.0;
    let sensors = vec![
        [0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, 0.0, 0.01],
        [-0.01, 0.0, 0.0],
    ];

    let config = TrilaterationConfig {
        sound_speed: c,
        initial_guess: Some([0.0, 0.0, 0.0]),
        ..Default::default()
    };
    let trilat = Trilateration::new(sensors.clone(), config).unwrap();

    let source_pos = [0.0, 0.0, 0.0];
    let arrival_times: Vec<f64> = sensors
        .iter()
        .map(|s| {
            let dx = source_pos[0] - s[0];
            let dy = source_pos[1] - s[1];
            let dz = source_pos[2] - s[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            dist / c
        })
        .collect();

    let result = trilat.localize(&arrival_times).unwrap();

    assert!(result.converged);
    assert_relative_eq!(result.position[0], 0.0, epsilon = 1e-4);
    assert_relative_eq!(result.position[1], 0.0, epsilon = 1e-4);
    assert_relative_eq!(result.position[2], 0.0, epsilon = 1e-4);
}

#[test]
fn test_localize_off_axis_source() {
    let c = 1540.0;
    let sensors = vec![
        [0.0, 0.0, 0.0],
        [0.02, 0.0, 0.0],
        [0.0, 0.02, 0.0],
        [0.0, 0.0, 0.02],
        [0.01, 0.01, 0.01],
    ];

    let config = TrilaterationConfig {
        sound_speed: c,
        ..Default::default()
    };
    let trilat = Trilateration::new(sensors.clone(), config).unwrap();

    let source_pos = [0.008, 0.008, 0.008];
    let arrival_times: Vec<f64> = sensors
        .iter()
        .map(|s| {
            let dx = source_pos[0] - s[0];
            let dy = source_pos[1] - s[1];
            let dz = source_pos[2] - s[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            dist / c
        })
        .collect();

    let result = trilat.localize(&arrival_times).unwrap();

    assert!(result.converged);
    assert_relative_eq!(result.position[0], source_pos[0], epsilon = 1e-4);
    assert_relative_eq!(result.position[1], source_pos[1], epsilon = 1e-4);
    assert_relative_eq!(result.position[2], source_pos[2], epsilon = 1e-4);
    assert!(result.residual < 1e-5);
}
