use super::super::{FwiGeometry, FwiProcessor};
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_domain::grid::Grid;
use kwavers_domain::source::{GridSource, SourceMode};
use crate::inverse::seismic::parameters::FwiParameters;
use ndarray::{Array2, Array3};

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
    let model = Array3::from_elem((3, 3, 3), SOUND_SPEED_WATER_SIM);

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
    let model = Array3::from_elem((3, 3, 3), SOUND_SPEED_WATER_SIM);

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
