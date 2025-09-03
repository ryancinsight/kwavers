//! Integration tests using plugin-based solver

use kwavers::{
    boundary::PMLBoundary,
    grid::Grid,
    medium::{CoreMedium, HomogeneousMedium},
    physics::constants::{DENSITY_WATER, SOUND_SPEED_WATER},
    solver::plugin_based::PluginBasedSolver,
    source::PointSource,
    time::Time,
};
use std::sync::Arc;

#[test]
fn test_point_source_propagation() {
    // Create grid
    let grid = Grid::new(64, 64, 64, 0.001, 0.001, 0.001);

    // Create medium
    let medium = HomogeneousMedium::new(SOUND_SPEED_WATER, DENSITY_WATER);

    // Create boundary
    let boundary = PMLBoundary::new(10);

    // Create source
    let source = PointSource::new((32, 32, 32), 1.0, 1e6, 0.0, 0.0);

    // Create time settings
    let time = Time::new(1e-7, 100); // Small timestep, 100 steps

    // Create solver
    let mut solver = PluginBasedSolver::new(
        grid,
        time,
        Arc::new(medium),
        Box::new(boundary),
        Box::new(source),
    );

    // Initialize
    solver.initialize().expect("Failed to initialize solver");

    // Run simulation
    solver.run_for_steps(50).expect("Failed to run simulation");

    // Verify solver ran
    assert!(solver.current_step >= 50, "Solver should have run 50 steps");
}

#[test]
fn test_grid_creation() {
    let grid = Grid::new(100, 100, 100, 0.001, 0.001, 0.001);
    assert_eq!(grid.nx, 100);
    assert_eq!(grid.ny, 100);
    assert_eq!(grid.nz, 100);
    assert_eq!(grid.total_points(), 1_000_000);
}

#[test]
fn test_medium_properties() {
    let medium = HomogeneousMedium::new(SOUND_SPEED_WATER, DENSITY_WATER);

    // Test at various points
    assert_eq!(medium.sound_speed(0, 0, 0), SOUND_SPEED_WATER);
    assert_eq!(medium.density(0, 0, 0), DENSITY_WATER);
    assert!(medium.is_homogeneous());
}
