#![cfg_attr(test, expect(clippy::unwrap_used, reason = "ratchet KWAVERS-UNWRAP-1"))]

use aequitas::systems::si::quantities::{Frequency, Length, Velocity};
use aequitas::systems::si::units::{Hertz, Meter, MeterPerSecond};
use kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE;

use super::*;

#[test]
fn test_mode_default() {
    assert_eq!(
        NeuralBeamformingMode::default(),
        NeuralBeamformingMode::Hybrid
    );
}

#[test]
fn test_physics_parameters_default() {
    let params = NeuralBeamformingPhysicsParams::default();
    assert_eq!(params.reciprocity_weight, 1.0);
    assert_eq!(params.coherence_weight, 0.5);
    assert_eq!(params.sparsity_weight, 0.1);
}

#[test]
fn test_sensor_geometry_linear() {
    let geometry = SensorGeometry::linear_array(
        64,
        Length::from_unit::<Meter>(0.0003),
        Frequency::from_unit::<Hertz>(40e6),
        Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_TISSUE),
    )
    .expect("valid linear sensor geometry");
    assert_eq!(geometry.num_elements(), 64);
    assert_eq!(geometry.sampling_frequency.in_unit::<Hertz>(), 40e6);

    assert!(
        (geometry.positions[31][0].in_unit::<Meter>()
            + geometry.positions[32][0].in_unit::<Meter>())
        .abs()
            < 1e-10
    );
}

#[test]
fn test_sensor_geometry_phased() {
    let geometry = SensorGeometry::phased_array(
        8,
        8,
        Length::from_unit::<Meter>(0.0003),
        Length::from_unit::<Meter>(0.0003),
        Frequency::from_unit::<Hertz>(40e6),
        Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_TISSUE),
    )
    .expect("valid phased sensor geometry");
    assert_eq!(geometry.num_elements(), 64);
}

#[test]
fn sensor_geometry_rejects_invalid_dimensions_and_parameters() {
    let frequency = Frequency::from_unit::<Hertz>(40e6);
    let sound_speed = Velocity::from_unit::<MeterPerSecond>(SOUND_SPEED_TISSUE);
    let zero_elements = SensorGeometry::linear_array(
        0,
        Length::from_unit::<Meter>(0.0003),
        frequency,
        sound_speed,
    );
    assert!(zero_elements.is_err());

    let zero_pitch =
        SensorGeometry::linear_array(8, Length::from_unit::<Meter>(0.0), frequency, sound_speed);
    assert!(zero_pitch.is_err());

    let nonfinite_frequency = SensorGeometry::linear_array(
        8,
        Length::from_unit::<Meter>(0.0003),
        Frequency::from_unit::<Hertz>(f64::NAN),
        sound_speed,
    );
    assert!(nonfinite_frequency.is_err());
}

#[test]
fn test_config_validation_valid() {
    // default: Hybrid mode, non-empty architecture, valid weights
    let config = NeuralBeamformingConfig::default();
    config.validate().unwrap();
    assert!(
        !config.network_architecture.is_empty(),
        "default must have non-empty network_architecture"
    );
    assert!(
        config.physics_parameters.reciprocity_weight > 0.0,
        "default reciprocity_weight must be positive"
    );
}

#[test]
fn test_config_validation_invalid_architecture() {
    let config = NeuralBeamformingConfig {
        network_architecture: vec![5],
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validation_invalid_physics_weight() {
    let mut config = NeuralBeamformingConfig::default();
    config.physics_parameters.reciprocity_weight = -1.0;
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validation_invalid_learning_rate() {
    let mut config = NeuralBeamformingConfig::default();
    config.adaptation_parameters.learning_rate = 0.0;
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validation_invalid_batch_size() {
    let config = NeuralBeamformingConfig {
        batch_size: 0,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validation_invalid_sensor_count() {
    let mut config = NeuralBeamformingConfig::default();
    config.sensor_geometry.positions = vec![[Length::from_unit::<Meter>(0.0); 3]];
    assert!(config.validate().is_err());
}
