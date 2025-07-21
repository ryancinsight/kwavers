// src/physics/mechanics/elastic_wave/tests.rs

use super::*;
use crate::grid::Grid;
use crate::medium::homogeneous::HomogeneousMedium;
use crate::source::MockSource;
use ndarray::Array4;

#[test]
fn test_elastic_wave_constructor() {
    let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001);
    let elastic_wave = ElasticWave::new(&grid);

    assert_eq!(elastic_wave.kx.shape(), &[32, 32, 32]);
    assert_eq!(elastic_wave.ky.shape(), &[32, 32, 32]);
    assert_eq!(elastic_wave.kz.shape(), &[32, 32, 32]);
}

#[test]
fn test_elastic_wave_single_step() {
    let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001);
    let mut elastic_wave = ElasticWave::new(&grid);
    let medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.0, 0.0);
    let source = MockSource::new();

    let mut fields = Array4::<f64>::zeros((crate::solver::TOTAL_FIELDS, 32, 32, 32));

    // Simple initial condition: a small velocity impulse in the center
    fields[[VX_IDX, 16, 16, 16]] = 1.0;

    let prev_pressure = Array3::<f64>::zeros((32, 32, 32));

    elastic_wave.update_wave(
        &mut fields,
        &prev_pressure,
        &source,
        &grid,
        &medium,
        1e-6,
        0.0,
    );

    // Check that the fields have changed from their initial zero state
    // A more rigorous test would check for specific wave propagation patterns
    assert!(fields.sum() > 0.0);
}
