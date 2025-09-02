//! Fixed integration tests using proper APIs

use kwavers::{
    grid::Grid,
    medium::HomogeneousMedium,
    signal::{ContinuousWave, Signal},
    solver::fdtd::{FdtdConfig, FdtdSolver},
    source::{PointSource, Source},
};
use ndarray::Array3;
use std::sync::Arc;

#[test]
fn test_point_source_propagation() {
    // Create grid
    let grid = Grid::new(64, 64, 64, 0.001, 0.001, 0.001);

    // Create medium
    let medium = HomogeneousMedium::water(&grid);

    // Create solver
    let config = FdtdConfig::default();
    let mut solver = FdtdSolver::new(config, &grid).unwrap();

    // Create source
    let frequency = 1e6; // 1 MHz
    let signal = Arc::new(ContinuousWave::new(frequency, 1.0, 0.0));
    let source = PointSource::new((0.032, 0.032, 0.032), signal);
    let source_mask = source.create_mask(&grid);

    // Initialize fields
    let mut pressure = Array3::zeros((grid.nx, grid.ny, grid.nz));
    let mut velocity_x = Array3::zeros((grid.nx, grid.ny, grid.nz));
    let mut velocity_y = Array3::zeros((grid.nx, grid.ny, grid.nz));
    let mut velocity_z = Array3::zeros((grid.nx, grid.ny, grid.nz));

    // Calculate stable time step
    let c_max = medium.sound_speed(0.0, 0.0, 0.0, &grid);
    let dt = solver.max_stable_dt(c_max);

    // Run simulation
    for step in 0..100 {
        let t = step as f64 * dt;

        // Apply source
        let amplitude = source.amplitude(t);
        pressure = &pressure + &source_mask * amplitude * dt;

        // Update fields (simplified - actual FDTD would be more complex)
        solver
            .update_pressure(
                &mut pressure,
                &velocity_x,
                &velocity_y,
                &velocity_z,
                &medium,
                &grid,
                dt,
            )
            .unwrap();

        solver
            .update_velocity(
                &mut velocity_x,
                &mut velocity_y,
                &mut velocity_z,
                &pressure,
                &medium,
                &grid,
                dt,
            )
            .unwrap();
    }

    // Basic verification - pressure should have propagated
    let center_pressure = pressure[[32, 32, 32]];
    assert!(
        center_pressure.abs() > 0.0,
        "No pressure at source location"
    );

    // Check that wave has spread
    let nearby_pressure = pressure[[35, 32, 32]];
    assert!(nearby_pressure.abs() > 0.0, "Wave didn't propagate");
}

#[test]
fn test_cfl_stability() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
    let config = FdtdConfig::default();
    let solver = FdtdSolver::new(config, &grid).unwrap();

    let c_max = 1500.0; // Water
    let dt = solver.max_stable_dt(c_max);

    // Should satisfy CFL
    assert!(solver.check_cfl_stability(dt, c_max));

    // Too large dt should violate CFL
    assert!(!solver.check_cfl_stability(dt * 2.0, c_max));
}
