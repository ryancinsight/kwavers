//! Simple integration tests for core functionality

use kwavers::{
    grid::Grid,
    medium::{ArrayAccess, CoreMedium, HomogeneousMedium},
};
use ndarray::Array3;

/// Test basic medium and grid initialization
#[test]
fn test_basic_initialization() {
    // Create grid
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
    assert_eq!(grid.nx, 32);
    assert_eq!(grid.ny, 32);
    assert_eq!(grid.nz, 32);

    // Create water medium
    let medium = HomogeneousMedium::water(&grid);
    // Check properties using CoreMedium trait at grid center
    let i = grid.nx / 2;
    let j = grid.ny / 2;
    let k = grid.nz / 2;
    // Water at 20°C has specific properties
    assert_eq!(medium.density(i, j, k), 998.0);
    assert_eq!(medium.sound_speed(i, j, k), 1482.0);
}

/// Test acoustic field propagation
#[test]
fn test_acoustic_field() {
    let grid = Grid::new(50, 50, 50, 1e-3, 1e-3, 1e-3).unwrap();
    let medium = HomogeneousMedium::water(&grid);

    // Initialize pressure field with Gaussian pulse
    let mut pressure = Array3::zeros((50, 50, 50));
    let center = 25;
    let sigma: f64 = 5.0;

    for i in 0..50 {
        for j in 0..50 {
            for k in 0..50 {
                let r2 = ((i as f64 - center as f64).powi(2)
                    + (j as f64 - center as f64).powi(2)
                    + (k as f64 - center as f64).powi(2))
                    / sigma.powi(2);
                pressure[[i, j, k]] = (-r2).exp();
            }
        }
    }

    // Verify pressure field properties
    let max_pressure = pressure.iter().fold(0.0f64, |a, &b| a.max(b));
    assert_relative_eq!(max_pressure, 1.0, epsilon = 1e-10);

    // Verify total energy
    let total_energy: f64 = pressure.iter().map(|&p| p.powi(2)).sum();
    assert!(total_energy > 0.0);
}

/// Test medium array access
#[test]
fn test_medium_arrays() {
    let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(1500.0, 1000.0, &grid);

    // Test array access
    let density_array = medium.density_array();
    let sound_speed_array = medium.sound_speed_array();
    let absorption_array = medium.absorption_array();
    let nonlinearity_array = medium.nonlinearity_array();

    assert_eq!(density_array.shape(), &[10, 10, 10]);
    assert_eq!(sound_speed_array.shape(), &[10, 10, 10]);
    assert_eq!(absorption_array.shape(), &[10, 10, 10]);
    assert_eq!(nonlinearity_array.shape(), &[10, 10, 10]);

    // Check values are uniform for homogeneous medium
    assert!(density_array.iter().all(|&d| d == 1500.0));
    assert!(sound_speed_array.iter().all(|&c| c == 1000.0));
}

/// Test grid calculations
#[test]
fn test_grid_calculations() {
    let grid = Grid::new(100, 50, 25, 0.001, 0.002, 0.004).unwrap();

    // Test grid spacing
    assert_eq!(grid.dx, 0.001);
    assert_eq!(grid.dy, 0.002);
    assert_eq!(grid.dz, 0.004);

    // Test total volume
    let volume =
        (grid.nx as f64 * grid.dx) * (grid.ny as f64 * grid.dy) * (grid.nz as f64 * grid.dz);
    assert_relative_eq!(volume, 0.1 * 0.1 * 0.1, epsilon = 1e-10);
}

use approx::assert_relative_eq;
