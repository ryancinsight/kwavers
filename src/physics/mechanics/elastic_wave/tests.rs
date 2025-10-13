// src/physics/mechanics/elastic_wave/tests.rs

#[cfg(test)]
use crate::grid::Grid;
#[cfg(test)]
use crate::medium::homogeneous::HomogeneousMedium;
#[cfg(test)]
use crate::physics::field_mapping::UnifiedFieldType;
#[cfg(test)]
use crate::physics::mechanics::elastic_wave::ElasticWave;
#[cfg(test)]
use crate::physics::traits::AcousticWaveModel;
#[cfg(test)]
use crate::source::NullSource;
#[cfg(test)]
use ndarray::{Array3, Array4};

#[test]
fn test_elastic_wave_constructor() {
    let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();
    let _elastic_wave = ElasticWave::new(&grid).unwrap();
    // Test that the constructor completes successfully - validated through successful execution
}

#[test]
fn test_elastic_wave_single_step() {
    let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();
    let mut elastic_wave = ElasticWave::new(&grid).unwrap();
    let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
    let source = NullSource::new();

    let mut fields = Array4::<f64>::zeros((crate::solver::TOTAL_FIELDS, 32, 32, 32));

    // Set initial conditions
    fields[[UnifiedFieldType::VelocityX.index(), 16, 16, 16]] = 1.0;

    let prev_pressure = Array3::<f64>::zeros((32, 32, 32));

    // This should run without panicking - validate result is handled
    let result = elastic_wave.update_wave(
        &mut fields,
        &prev_pressure,
        &source,
        &grid,
        &medium,
        1e-6,
        0.0,
    );
    // Test passes if update succeeds without error
    assert!(
        result.is_ok(),
        "Elastic wave update should succeed: {:?}",
        result.err()
    );
}
