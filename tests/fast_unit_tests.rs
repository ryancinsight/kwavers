//! Fast unit tests for SRS NFR-002 compliance (<30s execution)
//!
//! This module contains lightweight unit tests that execute quickly
//! to support continuous integration and rapid feedback cycles.
//! Heavy integration tests are separated into dedicated test files.

use kwavers::{
    error::GridError,
    grid::Grid,
    medium::{CoreMedium, HomogeneousMedium},
    physics::constants::{DENSITY_WATER, SOUND_SPEED_WATER},
};

#[test]
fn test_grid_creation_basic() {
    let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001);
    assert!(grid.is_ok());

    let grid = grid.unwrap();
    assert_eq!(grid.nx, 10);
    assert_eq!(grid.ny, 10);
    assert_eq!(grid.nz, 10);
    assert_eq!(grid.size(), 1000);
}

#[test]
fn test_grid_creation_validation() {
    // Test invalid dimensions
    let result = Grid::new(0, 10, 10, 0.001, 0.001, 0.001);
    assert!(result.is_err());

    // Test invalid spacing
    let result = Grid::new(10, 10, 10, 0.0, 0.001, 0.001);
    assert!(result.is_err());
}

#[test]
fn test_medium_properties_basic() {
    let grid = Grid::new(5, 5, 5, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(DENSITY_WATER, SOUND_SPEED_WATER, 0.0, 0.0, &grid);

    // Test homogeneity
    assert!(medium.is_homogeneous());

    // Test property access
    let sound_speed = medium.sound_speed(0, 0, 0);
    let density = medium.density(0, 0, 0);

    assert!((sound_speed - SOUND_SPEED_WATER).abs() < 1e-10);
    assert!((density - DENSITY_WATER).abs() < 1e-10);
}

#[test]
fn test_constants_validity() {
    // Physical constants are compile-time verified through const definitions
    // Runtime validation ensures constants are accessible and have expected types
    let density: f64 = DENSITY_WATER;
    let speed: f64 = SOUND_SPEED_WATER;

    // Validate through type constraints and reasonable value checks
    assert!(density > 0.0, "Density must be positive");
    assert!(speed > 0.0, "Sound speed must be positive");

    // Verify constants are within physically reasonable ranges (not constant evaluation)
    let density_valid = density > 900.0 && density < 1100.0;
    let speed_valid = speed > 1400.0 && speed < 1600.0;
    assert!(density_valid, "Water density should be ~1000 kg/m³");
    assert!(speed_valid, "Water sound speed should be ~1500 m/s");
}

#[test]
fn test_error_handling_basic() {
    // Test error propagation without heavy computation
    let invalid_grid = Grid::new(0, 0, 0, 0.0, 0.0, 0.0);
    assert!(invalid_grid.is_err());

    match invalid_grid {
        Err(GridError::ZeroDimension { .. }) => {} // Expected
        _ => panic!("Unexpected error type"),
    }
}

#[cfg(test)]
mod performance_unit_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_grid_creation_performance() {
        let start = Instant::now();
        let _grid = Grid::new(50, 50, 50, 0.001, 0.001, 0.001).unwrap();
        let duration = start.elapsed();

        // Grid creation should be very fast
        assert!(
            duration.as_millis() < 10,
            "Grid creation too slow: {:?}",
            duration
        );
    }

    #[test]
    fn test_medium_creation_performance() {
        let grid = Grid::new(20, 20, 20, 0.001, 0.001, 0.001).unwrap();
        let start = Instant::now();
        let _medium = HomogeneousMedium::new(DENSITY_WATER, SOUND_SPEED_WATER, 0.0, 0.0, &grid);
        let duration = start.elapsed();

        // Medium creation should be very fast
        assert!(
            duration.as_millis() < 5,
            "Medium creation too slow: {:?}",
            duration
        );
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;

    #[test]
    fn test_grid_dimensions_regression() {
        // Regression test for grid dimension calculation
        let grid = Grid::new(100, 200, 300, 0.001, 0.002, 0.003).unwrap();
        assert_eq!(grid.size(), 100 * 200 * 300);
        assert_eq!(grid.nx, 100);
        assert_eq!(grid.ny, 200);
        assert_eq!(grid.nz, 300);
    }

    #[test]
    fn test_medium_homogeneity_regression() {
        // Regression test for medium homogeneity detection
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);

        // Properties should be identical at all points for homogeneous medium
        let speed_center = medium.sound_speed(5, 5, 5);
        let speed_corner = medium.sound_speed(0, 0, 0);
        let speed_edge = medium.sound_speed(9, 9, 9);

        assert!((speed_center - speed_corner).abs() < 1e-15);
        assert!((speed_corner - speed_edge).abs() < 1e-15);
    }
}
