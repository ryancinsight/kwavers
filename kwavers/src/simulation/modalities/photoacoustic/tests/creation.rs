//! Simulator creation, optical property, and accessor tests.

use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use super::super::core::PhotoacousticSimulator;
use crate::domain::grid::Grid;
use crate::domain::imaging::photoacoustic::PhotoacousticOpticalProperties;
use crate::domain::medium::homogeneous::HomogeneousMedium;

#[test]
fn test_photoacoustic_creation() {
    let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(1000.0, SOUND_SPEED_WATER_SIM, 0.5, 1.0, &grid);
    let parameters = crate::domain::imaging::photoacoustic::PhotoacousticParameters::default();

    let _simulator = PhotoacousticSimulator::new(grid, parameters, &medium).unwrap();
}

#[test]
fn test_optical_properties() {
    let blood_props = PhotoacousticOpticalProperties::blood(750.0);
    let tissue_props = PhotoacousticOpticalProperties::soft_tissue(750.0);
    let tumor_props = PhotoacousticOpticalProperties::tumor(750.0);

    assert!(
        blood_props.absorption_coefficient > tissue_props.absorption_coefficient,
        "Blood has higher absorption due to hemoglobin"
    );
    assert!(
        tumor_props.absorption_coefficient > tissue_props.absorption_coefficient,
        "Tumors have higher absorption due to increased vascularity"
    );
    assert!(blood_props.absorption_coefficient > 0.0);
    assert!(tissue_props.absorption_coefficient > 0.0);
    assert!(tumor_props.absorption_coefficient > 0.0);
}

#[test]
fn test_accessor_methods() {
    let grid = Grid::new(16, 16, 8, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(1000.0, SOUND_SPEED_WATER_SIM, 0.5, 1.0, &grid);
    let parameters = crate::domain::imaging::photoacoustic::PhotoacousticParameters::default();
    let simulator = PhotoacousticSimulator::new(grid.clone(), parameters.clone(), &medium).unwrap();

    assert_eq!(simulator.grid().dimensions(), (16, 16, 8));
    assert_eq!(simulator.optical_properties().dim(), (16, 16, 8));
    assert_eq!(
        simulator.parameters().wavelengths.len(),
        parameters.wavelengths.len()
    );
}
