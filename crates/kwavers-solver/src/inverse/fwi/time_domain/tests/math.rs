use super::super::{FwiGeometry, FwiProcessor};
use kwavers_domain::source::{GridSource, SourceMode};
use crate::inverse::seismic::parameters::FwiParameters;
use ndarray::{Array2, Array3, Array4};

#[test]
fn test_gradient_calculation() {
    let processor = FwiProcessor::default();

    let forward_field = Array3::ones((10, 10, 10));
    let adjoint_field = Array3::from_elem((10, 10, 10), 2.0);

    let gradient = processor.calculate_interaction(&forward_field, &adjoint_field);

    // Expected: -1.0 * 2.0 = -2.0 (after smoothing, close to -2.0)
    assert!((gradient[[5, 5, 5]] + 2.0).abs() < 0.1);
}

#[test]
fn test_l2_adjoint_source_computation() {
    let processor = FwiProcessor::default();
    let observed = Array2::from_shape_vec((2, 3), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        .expect("shape must be valid");
    let synthetic = Array2::from_shape_vec((2, 3), vec![1.0, 0.5, 3.0, 1.0, 7.0, 9.0])
        .expect("shape must be valid");

    let adjoint_source = processor
        .compute_adjoint_source(&observed, &synthetic)
        .expect("adjoint source computation must succeed");

    let expected = Array2::from_shape_vec((2, 3), vec![1.0, -0.5, 1.0, -2.0, 3.0, 4.0])
        .expect("shape must be valid");
    assert_eq!(adjoint_source, expected);
}

#[test]
fn test_l2_objective_matches_definition() {
    let processor = FwiProcessor::new(FwiParameters {
        nt: 3,
        dt: 0.5,
        max_iterations: 1,
        step_size: 1.0,
        ..FwiParameters::default()
    });

    let observed =
        Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0]).expect("shape must be valid");
    let synthetic =
        Array2::from_shape_vec((2, 2), vec![2.0, 4.0, 6.0, 8.0]).expect("shape must be valid");

    let objective = processor
        .compute_l2_objective(&observed, &synthetic)
        .expect("objective computation must succeed");

    // residual = [1,3,5,7], sum(residual^2) = 84, objective = 0.5 * dt * 84 = 21
    assert!((objective - 21.0).abs() < f64::EPSILON);
}

#[test]
fn test_adjoint_source_reorders_and_time_reverses() {
    let processor = FwiProcessor::new(FwiParameters {
        nt: 3,
        dt: 1.0,
        max_iterations: 1,
        step_size: 1.0,
        ..FwiParameters::default()
    });

    let sensor_mask = Array3::from_shape_vec((2, 2, 1), vec![true, true, true, true])
        .expect("shape must be valid");
    let geometry = FwiGeometry::new(GridSource::default(), sensor_mask);

    let residual = Array2::from_shape_vec(
        (4, 3),
        vec![
            1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 100.0, 200.0, 300.0, 1000.0, 2000.0, 3000.0,
        ],
    )
    .expect("shape must be valid");

    let source = processor
        .build_adjoint_source(&residual, &geometry)
        .expect("adjoint source construction must succeed");

    let GridSource {
        p_mask,
        p_signal,
        p_mode,
        ..
    } = source;
    let p_signal = p_signal.expect("pressure signal must be present");
    let expected = Array2::from_shape_vec(
        (4, 3),
        vec![
            3.0, 2.0, 1.0, 300.0, 200.0, 100.0, 30.0, 20.0, 10.0, 3000.0, 2000.0, 1000.0,
        ],
    )
    .expect("shape must be valid");

    assert_eq!(p_signal, expected);

    let p_mask = p_mask.expect("pressure mask must be present");
    assert_eq!(
        p_mask,
        geometry
            .sensor_mask
            .clone()
            .mapv(|active| if active { 1.0 } else { 0.0 })
    );
    assert!(matches!(p_mode, SourceMode::Additive));
}

#[test]
fn test_pressure_second_derivative_exact_for_quadratic_trace() {
    let processor = FwiProcessor::new(FwiParameters {
        nt: 5,
        dt: 1.0,
        max_iterations: 1,
        step_size: 1.0,
        ..FwiParameters::default()
    });

    let mut forward_history = Array4::zeros((5, 1, 1, 1));
    for t in 0..5 {
        forward_history[[t, 0, 0, 0]] = (t as f64).powi(2);
    }

    let mut dst = Array3::zeros((1, 1, 1));
    for idx in 0..5 {
        processor
            .pressure_second_derivative_into(&forward_history, idx, 1.0, &mut dst)
            .expect("second derivative computation must succeed");
        assert!((dst[[0, 0, 0]] - 2.0).abs() < f64::EPSILON);
    }
}
