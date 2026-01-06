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
    let grid = Grid::new(64, 64, 64, 0.001, 0.001, 0.001).expect("Failed to create grid");

    // Create medium
    let medium = HomogeneousMedium::new(DENSITY_WATER, SOUND_SPEED_WATER, 0.0, 0.0, &grid);

    // Create boundary
    let pml_config = kwavers::boundary::PMLConfig {
        thickness: 10,
        ..Default::default()
    };
    let boundary = PMLBoundary::new(pml_config).expect("Failed to create PML boundary");

    // Create source
    let source = PointSource::new(
        (32.0, 32.0, 32.0),
        Arc::new(kwavers::signal::SineWave::new(1e6, 1.0, 0.0)),
    );

    // Create time settings
    let time = Time::new(1e-7, 100); // Small timestep, 100 steps

    // Create solver
    let mut solver = PluginBasedSolver::new(
        grid.clone(),
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
    // Note: current_step is private, so we can't directly test it
    // The test passes if the solver runs without panicking
}

#[test]
fn test_grid_creation() {
    let grid = Grid::new(100, 100, 100, 0.001, 0.001, 0.001).expect("Failed to create grid");
    assert_eq!(grid.nx, 100);
    assert_eq!(grid.ny, 100);
    assert_eq!(grid.nz, 100);
    assert_eq!(grid.size(), 1_000_000);
}

#[test]
fn test_medium_properties() {
    let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).expect("Failed to create grid");
    let medium = HomogeneousMedium::new(DENSITY_WATER, SOUND_SPEED_WATER, 0.0, 0.0, &grid);

    // Test at various points - use approximate equality for floating point
    assert!(
        (medium.sound_speed(0, 0, 0) - SOUND_SPEED_WATER).abs() < 1e-6,
        "Sound speed mismatch: expected {}, got {}",
        SOUND_SPEED_WATER,
        medium.sound_speed(0, 0, 0)
    );
    assert!(
        (medium.density(0, 0, 0) - DENSITY_WATER).abs() < 1e-6,
        "Density mismatch: expected {}, got {}",
        DENSITY_WATER,
        medium.density(0, 0, 0)
    );
    assert!(medium.is_homogeneous());
}
