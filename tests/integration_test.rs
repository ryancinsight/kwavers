//! Integration tests for Kwavers
//! 
//! These tests verify that the library components work together correctly.

use kwavers::{Grid, Time};
use kwavers::medium::{
    core::CoreMedium,
    homogeneous::HomogeneousMedium,
};

#[test]
fn test_grid_creation() {
    // Test that grid creation works
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
    assert_eq!(grid.nx, 32);
    assert_eq!(grid.ny, 32);
    assert_eq!(grid.nz, 32);
    assert_eq!(grid.dx, 1e-3);
    assert_eq!(grid.dy, 1e-3);
    assert_eq!(grid.dz, 1e-3);
}



#[test]
fn test_time_creation() {
    let time = Time::new(1e-7, 100);
    assert_eq!(time.dt, 1e-7);
    assert_eq!(time.num_steps(), 100);
}

#[test]
fn test_medium_creation() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
    let medium = HomogeneousMedium::new(
        1000.0,  // density
        1500.0,  // sound speed
        1e-3,    // viscosity
        0.072,   // surface tension
        &grid
    );
    
    // Test that medium properties are set correctly
    use kwavers::medium::Medium;
    assert_eq!(medium.density(0.0, 0.0, 0.0, &grid), 1000.0);
    assert_eq!(medium.sound_speed(0.0, 0.0, 0.0, &grid), 1500.0);
    assert!(medium.is_homogeneous());
}

#[test]
fn test_grid_field_creation() {
    let grid = Grid::new(10, 20, 30, 1e-3, 1e-3, 1e-3);
    let field = grid.create_field();
    
    // Check dimensions
    assert_eq!(field.shape(), &[10, 20, 30]);
    
    // Check that it's zero-initialized
    let sum: f64 = field.iter().sum();
    assert_eq!(sum, 0.0);
}

#[test]
fn test_grid_position_to_indices() {
    let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3);
    
    // Test center position
    let indices = grid.position_to_indices(5e-3, 5e-3, 5e-3);
    assert_eq!(indices, Some((5, 5, 5)));
    
    // Test origin
    let indices = grid.position_to_indices(0.0, 0.0, 0.0);
    assert_eq!(indices, Some((0, 0, 0)));
    
    // Test out of bounds
    let indices = grid.position_to_indices(11e-3, 5e-3, 5e-3);
    assert_eq!(indices, None);
    
    // Test negative (out of bounds)
    let indices = grid.position_to_indices(-1e-3, 5e-3, 5e-3);
    assert_eq!(indices, None);
}

#[test]
fn test_cfl_calculation() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
    let sound_speed = 1500.0; // m/s
    let cfl = 0.5;
    
    // Calculate expected timestep
    let min_dx = grid.dx.min(grid.dy).min(grid.dz);
    let expected_dt = cfl * min_dx / sound_speed;
    
    // For 3D FDTD, CFL should be around 0.5
    assert!(expected_dt > 0.0);
    assert!(expected_dt < 1e-6); // Should be microseconds range
}

#[test]
fn test_field_operations() {
    let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3);
    let mut field = grid.create_field();
    
    // Set a value
    field[[5, 5, 5]] = 1.0;
    assert_eq!(field[[5, 5, 5]], 1.0);
    
    // Check that only one element is non-zero
    let count: usize = field.iter().filter(|&&x| x != 0.0).count();
    assert_eq!(count, 1);
}