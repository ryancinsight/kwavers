//! Full simulation pipeline and multi-wavelength simulation tests.

use super::super::core::PhotoacousticSimulator;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_grid::Grid;
use kwavers_medium::homogeneous::HomogeneousMedium;

#[test]
fn test_simulation() {
    let grid = Grid::new(16, 16, 8, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.5,
        1.0,
        &grid,
    );
    let parameters = kwavers_imaging::photoacoustic::PhotoacousticParameters::default();
    let mut simulator = PhotoacousticSimulator::new(grid, parameters, &medium).unwrap();

    let fluence = simulator.compute_fluence().unwrap();
    let initial_pressure = simulator.compute_initial_pressure(&fluence).unwrap();
    let sim_result = simulator.simulate(&initial_pressure).unwrap();

    assert_eq!(sim_result.pressure_fields.len(), sim_result.time.len());
    assert!(
        sim_result.pressure_fields.len() >= 2,
        "Should have multiple time snapshots"
    );
    assert_eq!(sim_result.reconstructed_image.shape(), [16, 16, 8]);
    assert!(sim_result.snr > 0.0);

    for field in sim_result.pressure_fields.iter() {
        assert_eq!(field.shape(), [16, 16, 8]);
        for &val in field.iter() {
            assert!(val.is_finite(), "Pressure field values must be finite");
        }
    }

    for &val in sim_result.reconstructed_image.iter() {
        assert!(val.is_finite(), "Reconstructed image values must be finite");
    }
}

#[test]
fn test_multi_wavelength_simulation() {
    let grid = Grid::new(8, 8, 4, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.5,
        1.0,
        &grid,
    );
    let parameters = kwavers_imaging::photoacoustic::PhotoacousticParameters {
        wavelengths: vec![700.0, 800.0],
        ..Default::default()
    };

    let simulator = PhotoacousticSimulator::new(grid, parameters, &medium).unwrap();

    let multi_results = simulator.simulate_multi_wavelength().unwrap();
    assert_eq!(multi_results.len(), 2, "Should simulate all wavelengths");

    for (fluence, pressure) in &multi_results {
        assert_eq!(fluence.shape(), [8, 8, 4]);
        assert_eq!(pressure.pressure.shape(), [8, 8, 4]);
        assert!(pressure.max_pressure > 0.0);
    }
}
