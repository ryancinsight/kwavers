//! Pressure-buffer identity, wave propagation, and conservation-diagnostic integration tests.

use crate::forward::nonlinear::conservation::{ConservationDiagnostics, ConservationTolerances};
use crate::forward::nonlinear::westervelt::{WesterveltFdtd, WesterveltFdtdConfig};
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;

#[test]
fn westervelt_update_reuses_pressure_and_laplacian_workspaces() {
    let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
    let medium =
        HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, &grid);
    let config = WesterveltFdtdConfig {
        spatial_order: 2,
        enable_absorption: false,
        artificial_viscosity: 0.0,
        ..WesterveltFdtdConfig::default()
    };
    let mut solver = WesterveltFdtd::new(config, &grid, &medium);
    solver.pressure[[4, 4, 4]] = 1.0e5;

    let current = solver.pressure.as_ptr();
    let previous = solver.pressure_prev.as_ptr();
    let output = solver.pressure_next.as_ptr();
    let laplacian = solver.laplacian.as_ptr();
    let dt = solver.calculate_dt(&medium, &grid).unwrap();
    solver.update(&medium, &grid, &[], 0.0, dt).unwrap();

    assert_eq!(solver.pressure.as_ptr(), output);
    assert_eq!(solver.pressure_prev.as_ptr(), current);
    assert_eq!(solver.pressure_next.as_ptr(), previous);
    assert_eq!(solver.laplacian.as_ptr(), laplacian);
}

#[test]
fn test_linear_wave_propagation() {
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3).unwrap();
    let mut medium =
        HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, &grid);
    medium.set_nonlinearity(0.0);

    let config = WesterveltFdtdConfig {
        artificial_viscosity: 0.0,
        ..WesterveltFdtdConfig::default()
    };
    let mut solver = WesterveltFdtd::new(config, &grid, &medium);

    let center = (grid.nx / 2, grid.ny / 2, grid.nz / 2);
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let r2 = ((i as i32 - center.0 as i32).pow(2)
                    + (j as i32 - center.1 as i32).pow(2)
                    + (k as i32 - center.2 as i32).pow(2)) as f64;
                solver.pressure[[i, j, k]] = (-(r2 / 100.0)).exp();
            }
        }
    }

    let dt = solver.calculate_dt(&medium, &grid).unwrap();
    for _ in 0..10 {
        solver.update(&medium, &grid, &[], 0.0, dt).unwrap();
    }

    let total_energy: f64 = solver.pressure.iter().map(|&p| p * p).sum();
    assert!(total_energy > 0.0);
}

#[test]
fn test_conservation_diagnostics_integration() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
    let medium =
        HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, &grid);
    let config = WesterveltFdtdConfig::default();
    let mut solver = WesterveltFdtd::new(config, &grid, &medium);

    solver.enable_conservation_diagnostics(ConservationTolerances::default());

    let initial_energy = solver.calculate_total_energy();
    assert!(initial_energy < 1e-10);

    assert_eq!(
        solver.conservation_tracker.as_ref().unwrap().initial_energy,
        0.0
    );
    assert!(solver.is_solution_valid());

    solver.disable_conservation_diagnostics();
    assert!(solver.conservation_tracker.is_none());
}

#[test]
fn mismatched_grid_leaves_an_active_step_retryable() {
    use kwavers_core::error::{GridError, KwaversError};
    let grid = Grid::new(8, 9, 10, 1e-3, 1e-3, 1e-3).unwrap();
    let transposed = Grid::new(10, 9, 8, grid.dx, grid.dy, grid.dz).unwrap();
    let medium = HomogeneousMedium::water(&grid);
    let build = || {
        let mut solver = WesterveltFdtd::new(Default::default(), &grid, &medium);
        solver.pressure[[4, 4, 4]] = 1.0e5;
        solver.update(&medium, &grid, &[], 0.0, 1e-7).unwrap();
        solver
    };
    let mut actual = build();
    let mut expected = build();
    let result = actual.update(&medium, &transposed, &[], 1e-7, 1e-7);
    assert!(matches!(
        result,
        Err(KwaversError::Grid(GridError::DimensionMismatch { .. }))
    ));
    assert_eq!(actual.current_step, expected.current_step);
    assert_eq!(actual.current_time, expected.current_time);
    for _ in 0..2 {
        assert_eq!(actual.pressure, expected.pressure);
        assert_eq!(actual.pressure_prev, expected.pressure_prev);
        assert_eq!(actual.pressure_next, expected.pressure_next);
        assert_eq!(actual.laplacian, expected.laplacian);
        actual
            .update(&medium, &grid, &[], actual.current_time, 1e-7)
            .unwrap();
        expected
            .update(&medium, &grid, &[], expected.current_time, 1e-7)
            .unwrap();
    }
}
