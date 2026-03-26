//! Tests for aberration correction

use super::*;
use crate::domain::grid::Grid;
// use crate::physics::skull::CTBasedSkullModel;
use ndarray::Array3;
use num_complex::Complex;

#[test]
fn test_aberration_corrector_creation() {
    let grid = Grid::new(32, 32, 32, 0.002, 0.002, 0.002).unwrap();
    let corrector = TranscranialAberrationCorrection::new(&grid);
    assert!(corrector.is_ok());
}

#[test]
fn test_phase_correction_calculation() {
    let grid = Grid::new(16, 16, 16, 0.005, 0.005, 0.005).unwrap();
    let corrector = TranscranialAberrationCorrection::new(&grid).unwrap();

    let ct_data = Array3::from_elem((16, 16, 16), 400.0);

    let transducer_positions = vec![[0.0, 0.0, 0.08], [0.02, 0.0, 0.08], [0.0, 0.02, 0.08]];

    let target_point = [0.0, 0.0, 0.0];

    let correction =
        corrector.calculate_correction(&ct_data, &transducer_positions, &target_point);

    assert!(correction.is_ok());
    let corr = correction.unwrap();
    assert_eq!(corr.phases.len(), transducer_positions.len());
    assert_eq!(corr.amplitudes.len(), transducer_positions.len());
}

#[test]
fn test_time_reversal_correction() {
    let grid = Grid::new(16, 16, 16, 0.005, 0.005, 0.005).unwrap();
    let corrector = TranscranialAberrationCorrection::new(&grid).unwrap();

    let transducer_positions = vec![[0.0, 0.0, 0.08], [0.02, 0.0, 0.08]];

    let measured_field = Array3::from_elem((16, 16, 16), Complex::new(1.0, 0.0));

    let correction =
        corrector.apply_time_reversal_correction(&measured_field, &transducer_positions);

    assert!(correction.is_ok());
}
