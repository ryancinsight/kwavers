//! Optical fluence computation tests (single- and multi-wavelength).

use super::super::core::PhotoacousticSimulator;
use crate::core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use crate::domain::grid::Grid;
use crate::domain::medium::homogeneous::HomogeneousMedium;

#[test]
fn test_fluence_computation() {
    let grid = Grid::new(16, 16, 8, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, 0.5, 1.0, &grid);
    let parameters = crate::domain::imaging::photoacoustic::PhotoacousticParameters::default();
    let simulator = PhotoacousticSimulator::new(grid, parameters, &medium).unwrap();

    let fluence_data = simulator.compute_fluence().unwrap();
    assert_eq!(fluence_data.dim(), (16, 16, 8));

    let surface_fluence = fluence_data[[8, 8, 0]];
    let deep_fluence = fluence_data[[8, 8, 7]];
    assert!(
        surface_fluence > deep_fluence,
        "Fluence should decrease with depth due to absorption and scattering"
    );

    for &val in fluence_data.iter() {
        assert!(val >= 0.0, "Fluence must be non-negative");
        assert!(val.is_finite(), "Fluence must be finite");
    }
}

#[test]
fn test_multi_wavelength_fluence() {
    let grid = Grid::new(8, 8, 4, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, 0.5, 1.0, &grid);
    let parameters = crate::domain::imaging::photoacoustic::PhotoacousticParameters {
        wavelengths: vec![700.0, 750.0, 800.0],
        ..Default::default()
    };

    let simulator = PhotoacousticSimulator::new(grid, parameters, &medium).unwrap();

    let fields = simulator.compute_multi_wavelength_fluence().unwrap();
    assert_eq!(
        fields.len(),
        3,
        "Should compute fluence for all wavelengths"
    );

    for field in &fields {
        assert_eq!(field.dim(), (8, 8, 4));
        for &val in field.iter() {
            assert!(val >= 0.0, "Fluence must be non-negative");
            assert!(val.is_finite(), "Fluence must be finite");
        }
    }
}
