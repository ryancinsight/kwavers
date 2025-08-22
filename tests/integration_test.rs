//! Integration tests for kwavers library
//! 
//! These tests demonstrate basic functionality works

use kwavers::{
    grid::Grid,
    medium::HomogeneousMedium,
    error::KwaversResult,
};

#[test]
fn test_grid_creation() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
    assert_eq!(grid.nx, 32);
    assert_eq!(grid.ny, 32);
    assert_eq!(grid.nz, 32);
    assert_eq!(grid.dx, 1e-3);
}

#[test]
fn test_medium_creation() {
    use kwavers::medium::Medium;
    
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
    let medium = HomogeneousMedium::water(&grid);
    
    // Water properties at 20°C - access through trait methods
    let density = medium.density(0.0, 0.0, 0.0, &grid);
    let sound_speed = medium.sound_speed(0.0, 0.0, 0.0, &grid);
    
    assert!((density - 998.0).abs() < 10.0);
    assert!((sound_speed - 1480.0).abs() < 100.0);
}

#[test]
fn test_cfl_timestep() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
    let sound_speed = 1500.0;
    let cfl_factor = 0.95;
    
    let dt = grid.cfl_timestep(sound_speed, cfl_factor);
    
    // Check dt is positive and reasonable
    assert!(dt > 0.0);
    assert!(dt < 1e-3); // Should be less than spatial step / sound speed
}

#[test]
fn test_grid_field_creation() {
    let grid = Grid::new(16, 16, 16, 1e-3, 1e-3, 1e-3);
    let field = grid.create_field();
    
    assert_eq!(field.shape(), &[16, 16, 16]);
}

#[test]
fn test_library_builds() -> KwaversResult<()> {
    // This test just verifies the library compiles and basic types work
    let _grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3);
    let _medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &_grid);
    
    Ok(())
}