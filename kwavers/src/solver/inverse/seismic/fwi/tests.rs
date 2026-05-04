use super::*;
use crate::domain::grid::Grid;
use crate::domain::source::{GridSource, SourceMode};
use crate::solver::inverse::seismic::parameters::FwiParameters;
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

#[test]
fn test_forward_model_objective_vanishes_for_self_data() {
    let processor = FwiProcessor::new(FwiParameters {
        nt: 3,
        dt: 1e-4,
        max_iterations: 1,
        step_size: 1.0,
        ..FwiParameters::default()
    });

    let grid = Grid::new(3, 3, 3, 1.0, 1.0, 1.0).expect("grid must be valid");
    let model = Array3::from_elem((3, 3, 3), 1500.0);

    let mut p_mask = Array3::zeros((3, 3, 3));
    p_mask[[1, 1, 1]] = 1.0;
    let mut source = GridSource::default();
    source.p_mask = Some(p_mask);
    source.p_signal =
        Some(Array2::from_shape_vec((1, 3), vec![1.0, 0.0, 0.0]).expect("shape must be valid"));
    source.p_mode = SourceMode::Dirichlet;

    let mut sensor_mask = Array3::from_elem((3, 3, 3), false);
    sensor_mask[[2, 2, 2]] = true;
    let geometry = FwiGeometry::new(source, sensor_mask);

    let (synthetic, _history) = processor
        .forward_model(&model, &geometry, &grid)
        .expect("forward model must succeed");
    let objective = processor
        .compute_l2_objective(&synthetic, &synthetic)
        .expect("objective computation must succeed");

    assert!((objective - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_generate_synthetic_data_matches_canonical_forward_model() {
    let processor = FwiProcessor::new(FwiParameters {
        nt: 3,
        dt: 1e-4,
        max_iterations: 1,
        step_size: 1.0,
        ..FwiParameters::default()
    });

    let grid = Grid::new(3, 3, 3, 1.0, 1.0, 1.0).expect("grid must be valid");
    let model = Array3::from_elem((3, 3, 3), 1500.0);

    let mut p_mask = Array3::zeros((3, 3, 3));
    p_mask[[1, 1, 1]] = 1.0;
    let mut source = GridSource::default();
    source.p_mask = Some(p_mask);
    source.p_signal =
        Some(Array2::from_shape_vec((1, 3), vec![1.0, 0.0, 0.0]).expect("shape must be valid"));
    source.p_mode = SourceMode::Dirichlet;

    let mut sensor_mask = Array3::from_elem((3, 3, 3), false);
    sensor_mask[[2, 2, 2]] = true;
    let geometry = FwiGeometry::new(source, sensor_mask);

    let public_data = processor
        .generate_synthetic_data(&model, &geometry, &grid)
        .expect("public synthetic data generation must succeed");
    let (canonical_data, _history) = processor
        .forward_model(&model, &geometry, &grid)
        .expect("canonical forward model must succeed");

    assert_eq!(public_data, canonical_data);
    assert_eq!(public_data.dim(), (1, 3));
}

#[test]
fn test_model_constraints() {
    let processor = FwiProcessor::default();
    let mut model = Array3::from_elem((5, 5, 5), 10000.0);

    processor.apply_model_constraints(&mut model);

    assert!(model[[2, 2, 2]] <= 6000.0);
    assert!(model[[2, 2, 2]] >= 750.0);
}

/// Verify that the FWI forward-model medium is built with seismic (non-water) density.
#[test]
fn test_fwi_medium_density_not_water() {
    use crate::domain::medium::heterogeneous::HeterogeneousFactory;
    use crate::domain::medium::CoreMedium;

    let (nx, ny, nz) = (8usize, 8, 8);
    let sound_speed = Array3::from_elem((nx, ny, nz), 2000.0_f64);
    let density = Array3::from_elem((nx, ny, nz), RHO_SEISMIC_REF);

    let medium = HeterogeneousFactory::from_arrays(sound_speed, density, None, None, None, 20.0)
        .expect("medium construction must succeed");

    let rho_sample = medium.density(4, 4, 4);
    assert!(
        (rho_sample - RHO_SEISMIC_REF).abs() < 1.0,
        "medium density {rho_sample} != RHO_SEISMIC_REF {RHO_SEISMIC_REF}"
    );
    assert!(
        (rho_sample - 1000.0).abs() > 100.0,
        "density must not equal water (1000 kg/m³)"
    );
}

/// Verify that the FWI forward-model medium stores the velocity model correctly.
#[test]
fn test_fwi_forward_medium_sound_speed_matches_model() {
    use crate::domain::medium::heterogeneous::HeterogeneousFactory;
    use crate::domain::medium::CoreMedium;

    let (nx, ny, nz) = (6usize, 6, 6);
    let mut model = Array3::from_elem((nx, ny, nz), 1800.0_f64);
    model[[3, 3, 3]] = 3200.0;

    let density = Array3::from_elem((nx, ny, nz), RHO_SEISMIC_REF);
    let medium = HeterogeneousFactory::from_arrays(model.clone(), density, None, None, None, 20.0)
        .expect("medium construction must succeed");

    let c_bg = medium.sound_speed(1, 1, 1);
    let c_anom = medium.sound_speed(3, 3, 3);
    assert!((c_bg - 1800.0).abs() < 1.0, "background speed mismatch");
    assert!((c_anom - 3200.0).abs() < 1.0, "anomaly speed mismatch");
}
