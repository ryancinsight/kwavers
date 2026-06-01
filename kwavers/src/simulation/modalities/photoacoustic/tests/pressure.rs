//! Initial pressure computation and spherical spreading correction tests.

use super::super::core::PhotoacousticSimulator;
use crate::core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use crate::domain::grid::Grid;
use crate::domain::medium::homogeneous::HomogeneousMedium;
use ndarray::Array3;

#[test]
fn test_initial_pressure_computation() {
    let grid = Grid::new(16, 16, 8, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.5,
        1.0,
        &grid,
    );
    let parameters = crate::domain::imaging::photoacoustic::PhotoacousticParameters::default();
    let simulator = PhotoacousticSimulator::new(grid, parameters, &medium).unwrap();

    let fluence = simulator.compute_fluence().unwrap();
    let pressure_data = simulator.compute_initial_pressure(&fluence).unwrap();
    assert_eq!(pressure_data.pressure.dim(), (16, 16, 8));
    assert!(pressure_data.max_pressure > 0.0);

    for &val in pressure_data.pressure.iter() {
        assert!(val >= 0.0, "Pressure must be non-negative");
        assert!(val.is_finite(), "Pressure must be finite");
    }
}

#[test]
fn test_spherical_spreading_correction() {
    let grid = Grid::new(16, 16, 8, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.5,
        1.0,
        &grid,
    );
    let parameters = crate::domain::imaging::photoacoustic::PhotoacousticParameters::default();
    let simulator = PhotoacousticSimulator::new(grid, parameters, &medium).unwrap();

    let mut pressure_fields = vec![Array3::<f64>::zeros((16, 16, 8))];
    pressure_fields[0].fill(1.0);
    let time_points = vec![0.0];

    let reconstructed = simulator
        .time_reversal_reconstruction(&pressure_fields, &time_points)
        .unwrap();

    assert_eq!(reconstructed.dim(), (16, 16, 8));

    let center_value = reconstructed[[8, 8, 4]];
    let edge_value = reconstructed[[0, 0, 0]];

    assert_ne!(
        center_value, edge_value,
        "Reconstruction should have spatial variation due to 1/r weighting"
    );

    for &val in reconstructed.iter() {
        assert!(val.is_finite(), "Reconstructed values must be finite");
        assert!(
            val >= 0.0,
            "Reconstructed values should be non-negative due to 1/r weighting"
        );
    }
}
