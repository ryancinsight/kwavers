//! Tests for photoacoustic acoustic pressure and propagation.

use super::pressure::{compute_initial_pressure, compute_multi_wavelength_pressure};
use super::propagation::propagate_acoustic_wave;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_core::constants::numerical::MPA_TO_PA;
use kwavers_core::constants::thermodynamic::GRUNEISEN_WATER_20C;
use kwavers_grid::Grid;
use kwavers_imaging::photoacoustic::InitialPressure;
use kwavers_medium::homogeneous::HomogeneousMedium;
use approx::assert_relative_eq;
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

    let optical_properties =
        crate::modalities::photoacoustic::optics::initialize_optical_properties(
            &grid, &medium,
        )
        .unwrap();

    let fluence = Array3::from_elem((16, 16, 8), 1e6);

    let initial_pressure = compute_initial_pressure(
        &grid,
        &optical_properties,
        &fluence,
        &[GRUNEISEN_WATER_20C],
        &[750.0],
    )
    .unwrap();

    assert_eq!(initial_pressure.pressure.dim(), (16, 16, 8));
    assert!(initial_pressure.max_pressure > 0.0);

    for &val in initial_pressure.pressure.iter() {
        assert!(val >= 0.0, "Pressure must be non-negative");
        assert!(val.is_finite(), "Pressure must be finite");
    }
}

#[test]
fn test_wavelength_dependent_gruneisen() {
    let grid = Grid::new(8, 8, 4, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.5,
        1.0,
        &grid,
    );

    let optical_properties =
        crate::modalities::photoacoustic::optics::initialize_optical_properties(
            &grid, &medium,
        )
        .unwrap();

    let fluence = Array3::from_elem((8, 8, 4), 1e6);

    let pressure_visible = compute_initial_pressure(
        &grid,
        &optical_properties,
        &fluence,
        &[GRUNEISEN_WATER_20C],
        &[550.0],
    )
    .unwrap();

    let pressure_nir = compute_initial_pressure(
        &grid,
        &optical_properties,
        &fluence,
        &[GRUNEISEN_WATER_20C],
        &[750.0],
    )
    .unwrap();

    assert!(
        pressure_visible.max_pressure > pressure_nir.max_pressure,
        "Visible wavelengths should have higher thermoelastic efficiency"
    );
}

#[test]
fn test_multi_wavelength_pressure() {
    let grid = Grid::new(8, 8, 4, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.5,
        1.0,
        &grid,
    );

    let optical_properties =
        crate::modalities::photoacoustic::optics::initialize_optical_properties(
            &grid, &medium,
        )
        .unwrap();

    let fluence_fields = vec![
        Array3::from_elem((8, 8, 4), 1e6),
        Array3::from_elem((8, 8, 4), 1.2e6),
    ];

    let pressures = compute_multi_wavelength_pressure(
        &grid,
        &optical_properties,
        &fluence_fields,
        &[GRUNEISEN_WATER_20C, GRUNEISEN_WATER_20C],
        &[700.0, 800.0],
    )
    .unwrap();

    assert_eq!(pressures.len(), 2);
    for pressure in &pressures {
        assert_eq!(pressure.pressure.dim(), (8, 8, 4));
        assert!(pressure.max_pressure > 0.0);
    }
}

#[test]
fn test_acoustic_wave_propagation() {
    let grid = Grid::new(16, 16, 8, 0.001, 0.001, 0.001).unwrap();

    let mut pressure = Array3::zeros((16, 16, 8));
    pressure[[8, 8, 4]] = MPA_TO_PA;

    let initial_pressure = InitialPressure {
        pressure: pressure.clone(),
        max_pressure: MPA_TO_PA,
        fluence: pressure.clone(),
    };

    let (pressure_fields, time_points) = propagate_acoustic_wave(
        &grid,
        &initial_pressure,
        SOUND_SPEED_WATER_SIM,
        0.3,
        100,
        10,
    )
    .unwrap();

    assert!(pressure_fields.len() >= 10);
    assert_eq!(pressure_fields.len(), time_points.len());

    let initial_energy: f64 = pressure_fields[0].iter().map(|&x| x * x).sum();
    let final_energy: f64 = pressure_fields.last().unwrap().iter().map(|&x| x * x).sum();

    let energy_ratio = final_energy / initial_energy;
    assert!(
        energy_ratio > 0.1 && energy_ratio < 10.0,
        "Energy should be approximately conserved"
    );
}

#[test]
fn test_cfl_condition() {
    let grid = Grid::new(8, 8, 4, 0.001, 0.001, 0.001).unwrap();

    let pressure = Array3::from_elem((8, 8, 4), 1e5);
    let initial_pressure = InitialPressure {
        pressure: pressure.clone(),
        max_pressure: 1e5,
        fluence: pressure.clone(),
    };

    let speed_of_sound = SOUND_SPEED_WATER_SIM;
    let cfl_factor = 0.3;
    let min_h = grid.dx.min(grid.dy).min(grid.dz);
    let expected_dt = cfl_factor * min_h / speed_of_sound;

    let (_, time_points) =
        propagate_acoustic_wave(&grid, &initial_pressure, speed_of_sound, cfl_factor, 20, 10)
            .unwrap();

    if time_points.len() >= 2 {
        let actual_dt = time_points[1] - time_points[0];
        let dt_ratio = actual_dt / expected_dt;
        assert_relative_eq!(dt_ratio, 10.0, epsilon = 0.1);
    }
}
