//! Pressure-buffer identity, wave propagation, and conservation-diagnostic integration tests.

use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use crate::domain::grid::Grid;
use crate::domain::medium::HomogeneousMedium;
use crate::solver::forward::nonlinear::conservation::{
    ConservationDiagnostics, ConservationTolerances,
};
use crate::solver::forward::nonlinear::westervelt::{WesterveltFdtd, WesterveltFdtdConfig};

#[test]
fn westervelt_update_reuses_pressure_and_nonlinear_workspaces() {
    let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(1000.0, SOUND_SPEED_WATER_SIM, &grid);
    let config = WesterveltFdtdConfig {
        spatial_order: 2,
        enable_absorption: false,
        artificial_viscosity: 0.0,
        ..WesterveltFdtdConfig::default()
    };
    let mut solver = WesterveltFdtd::new(config, &grid, &medium);
    solver.pressure[[4, 4, 4]] = 1.0e5;

    let mut pressure_buffers_before = [
        solver.pressure.as_ptr() as usize,
        solver.pressure_prev.as_ptr() as usize,
        solver.pressure_next.as_ptr() as usize,
    ];
    pressure_buffers_before.sort_unstable();
    let nonlinear_before = solver.nonlinear_term.as_ptr();

    let dt = solver.calculate_dt(&medium, &grid).unwrap();
    solver.update(&medium, &grid, &[], 0.0, dt).unwrap();

    let mut pressure_buffers_after = [
        solver.pressure.as_ptr() as usize,
        solver.pressure_prev.as_ptr() as usize,
        solver.pressure_next.as_ptr() as usize,
    ];
    pressure_buffers_after.sort_unstable();

    assert_eq!(pressure_buffers_after, pressure_buffers_before);
    assert_eq!(solver.nonlinear_term.as_ptr(), nonlinear_before);
}

#[test]
fn test_linear_wave_propagation() {
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3).unwrap();
    let mut medium = HomogeneousMedium::from_minimal(1000.0, SOUND_SPEED_WATER_SIM, &grid);
    medium.nonlinearity = 0.0;

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
    let medium = HomogeneousMedium::from_minimal(1000.0, SOUND_SPEED_WATER_SIM, &grid);
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
