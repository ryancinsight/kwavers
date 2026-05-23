use super::*;
use crate::core::constants::numerical::MPA_TO_PA;
use crate::core::error::{KwaversError, ValidationError};

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
    let config =
        BowlConfig::from_vertex_focus([0.0, 0.0, 0.16], [0.0, 0.0, 0.0], 0.32, 650.0e3, MPA_TO_PA);
    let bowl = BowlTransducer::with_axis_projection_bounds(config, -0.28, 0.98, 16).unwrap();
    let array = MultiBowlArray::from_bowls(vec![bowl]).unwrap();

    assert_eq!(array.bowls.len(), 1);
    assert_eq!(array.bowls[0].element_count(), 16);
    assert_eq!(array.amplitudes, vec![MPA_TO_PA]);
}

#[test]
fn zero_amplitude_bowl_generates_finite_zero_field() {
    let config = BowlConfig {
        amplitude: 0.0,
        apply_directivity: false,
        ..small_bowl_config()
    };
    let bowl = BowlTransducer::with_element_count(config, 1).unwrap();
    let array = MultiBowlArray::from_bowls(vec![bowl]).unwrap();
    let grid = crate::domain::grid::Grid::new(2, 2, 2, 0.004, 0.005, 0.006).unwrap();
    let source = array.generate_source(&grid, 0.37e-6).unwrap();

    for value in source {
        assert!(value.is_finite());
        assert_eq!(value, 0.0);
    }
}

#[test]
fn hamming_apodization_preserves_pressure_units() {
    let config = BowlConfig {
        amplitude: 2.0e5,
        phase: 0.17,
        apply_directivity: false,
        ..small_bowl_config()
    };
    let bowl_a = BowlTransducer::with_element_count(config.clone(), 1).unwrap();
    let mut shifted = config;
    shifted.center = [0.002, 0.0, -0.08];
    shifted.focus = [0.002, 0.0, 0.0];
    let bowl_b = BowlTransducer::with_element_count(shifted, 1).unwrap();
    let grid = crate::domain::grid::Grid::new(2, 2, 2, 0.004, 0.005, 0.006).unwrap();

    let mut array = MultiBowlArray::from_bowls(vec![bowl_a, bowl_b]).unwrap();
    let untapered = array.generate_source(&grid, 0.37e-6).unwrap();
    array.apply_apodization(ApodizationType::Hamming);
    let tapered = array.generate_source(&grid, 0.37e-6).unwrap();

    let weights = ApodizationType::Hamming.weights(2);
    for (amplitude, weight) in array.amplitudes.iter().zip(weights.iter().copied()) {
        let expected = 2.0e5 * weight;
        assert_close(*amplitude, expected);
    }
    for (actual, reference) in tapered.iter().zip(untapered.iter()) {
        let expected = weights[0] * *reference;
        assert_close(*actual, expected);
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
        radius_of_curvature: 0.08,
        diameter: 0.04,
        center: [0.0, 0.0, -0.08],
        focus: [0.0, 0.0, 0.0],
        frequency: 1.25e6,
        amplitude: 1.0e5,
        element_size: Some(0.01),
        ..Default::default()
    }
}
