use super::*;
use aequitas::systems::si::quantities::{Angle, Dimensionless, Frequency, Length, Pressure, Time};
use aequitas::systems::si::units::{Hertz, Meter, Pascal, Radian, Second};
use kwavers_core::constants::numerical::MPA_TO_PA;
use kwavers_core::error::{KwaversError, ValidationError};

fn length(value: f64) -> Length<f64> {
    Length::from_unit::<Meter>(value)
}

fn point(values: [f64; 3]) -> [Length<f64>; 3] {
    values.map(length)
}

fn frequency(value: f64) -> Frequency<f64> {
    Frequency::from_unit::<Hertz>(value)
}

fn pressure(value: f64) -> Pressure<f64> {
    Pressure::from_unit::<Pascal>(value)
}

fn angle(value: f64) -> Angle<f64> {
    Angle::from_unit::<Radian>(value)
}

fn time(value: f64) -> Time<f64> {
    Time::from_unit::<Second>(value)
}

#[test]
fn multi_bowl_rejects_empty_source_set() {
    let error = MultiBowlArray::new(Vec::new()).unwrap_err();
    match error {
        KwaversError::Validation(ValidationError::FieldValidation {
            field,
            value,
            constraint,
        }) => {
            assert_eq!(field, "bowl_count");
            assert_eq!(value, "0");
            assert_eq!(constraint, "must be at least one");
        }
        other => panic!("expected bowl_count validation error, got {other:?}"),
    }
}

#[test]
fn from_bowls_accepts_preconstructed_bounded_layouts() {
    let config = BowlConfig::from_vertex_focus(
        point([0.0, 0.0, 0.16]),
        point([0.0, 0.0, 0.0]),
        length(0.32),
        frequency(650.0e3),
        pressure(MPA_TO_PA),
    );
    let bowl = BowlTransducer::with_axis_projection_bounds(
        config,
        Dimensionless::from_base(-0.28),
        Dimensionless::from_base(0.98),
        16,
    )
    .unwrap();
    let array = MultiBowlArray::from_bowls(vec![bowl]).unwrap();

    assert_eq!(array.bowls.len(), 1);
    assert_eq!(array.bowls[0].element_count(), 16);
    assert_eq!(array.amplitudes, vec![pressure(MPA_TO_PA)]);
}

#[test]
fn zero_amplitude_bowl_generates_finite_zero_field() {
    let config = BowlConfig {
        amplitude: pressure(0.0),
        apply_directivity: false,
        ..small_bowl_config()
    };
    let bowl = BowlTransducer::with_element_count(config, 1).unwrap();
    let array = MultiBowlArray::from_bowls(vec![bowl]).unwrap();
    let grid = kwavers_grid::Grid::new(2, 2, 2, 0.004, 0.005, 0.006).unwrap();
    let source = array.generate_source(&grid, time(0.37e-6)).unwrap();

    for &value in source.iter() {
        assert!(value.is_finite());
        assert_eq!(value, 0.0);
    }
}

#[test]
fn hamming_apodization_preserves_pressure_units() {
    let config = BowlConfig {
        amplitude: pressure(2.0e5),
        phase: angle(0.17),
        apply_directivity: false,
        ..small_bowl_config()
    };
    let bowl_a = BowlTransducer::with_element_count(config.clone(), 1).unwrap();
    let mut shifted = config;
    shifted.center = point([0.002, 0.0, -0.08]);
    shifted.focus = point([0.002, 0.0, 0.0]);
    let bowl_b = BowlTransducer::with_element_count(shifted, 1).unwrap();
    let grid = kwavers_grid::Grid::new(2, 2, 2, 0.004, 0.005, 0.006).unwrap();

    let mut array = MultiBowlArray::from_bowls(vec![bowl_a, bowl_b]).unwrap();
    let untapered = array.generate_source(&grid, time(0.37e-6)).unwrap();
    array.apply_apodization(ApodizationType::Hamming);
    let tapered = array.generate_source(&grid, time(0.37e-6)).unwrap();

    let weights = ApodizationType::Hamming.weights(2);
    for (amplitude, weight) in array.amplitudes.iter().zip(weights.iter().copied()) {
        let expected = 2.0e5 * weight;
        assert_close(amplitude.in_unit::<Pascal>(), expected);
    }
    for (actual, reference) in tapered.iter().zip(untapered.iter()) {
        let expected = weights[0] * *reference;
        assert_close(*actual, expected);
    }
}

#[test]
fn beam_steering_retargets_focus_without_moving_elements() {
    let bowl = BowlTransducer::with_element_count(small_bowl_config(), 8).unwrap();
    let original_positions = bowl.element_positions().to_vec();
    let mut array = MultiBowlArray::from_bowls(vec![bowl]).unwrap();
    let focus = point([0.01, -0.02, 0.03]);

    array.set_beam_steering(focus).unwrap();

    let bowl = &array.bowls[0];
    assert_eq!(bowl.element_positions(), original_positions.as_slice());
    assert_eq!(
        bowl.config.focus.map(|value| value.in_unit::<Meter>()),
        [0.01, -0.02, 0.03]
    );
    for (position, normal) in bowl.element_positions().iter().zip(bowl.element_normals()) {
        let position = position.map(|value| value.in_unit::<Meter>());
        let direction = [0.01 - position[0], -0.02 - position[1], 0.03 - position[2]];
        let distance = (direction[0] * direction[0]
            + direction[1] * direction[1]
            + direction[2] * direction[2])
            .sqrt();
        for (actual, expected) in normal.iter().zip(direction.map(|value| value / distance)) {
            assert!((actual - expected).abs() < 1.0e-12);
        }
    }
}

fn assert_close(actual: f64, expected: f64) {
    let tolerance = 64.0 * f64::EPSILON * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual {actual}, expected {expected}, tolerance {tolerance}"
    );
}

fn small_bowl_config() -> BowlConfig {
    BowlConfig {
        radius_of_curvature: length(0.08),
        diameter: length(0.04),
        center: point([0.0, 0.0, -0.08]),
        focus: point([0.0, 0.0, 0.0]),
        frequency: frequency(1.25e6),
        amplitude: pressure(1.0e5),
        element_size: Some(length(0.01)),
        ..Default::default()
    }
}
