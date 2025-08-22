//! Common test helper functions
//!
//! This module consolidates test utilities to avoid duplication
//! and maintain SSOT (Single Source of Truth) principle.

use crate::grid::Grid;
use crate::medium::homogeneous::HomogeneousMedium;
use ndarray::Array3;

/// Create a standard test grid with specified dimensions
///
/// This function provides a consistent grid for testing across modules
pub fn create_test_grid(nx: usize, ny: usize, nz: usize) -> Grid {
    Grid::new(nx, ny, nz, 0.001, 0.001, 0.001)
}

/// Create a standard test grid with default dimensions (32x32x32)
pub fn create_default_test_grid() -> Grid {
    create_test_grid(32, 32, 32)
}

/// Create a standard test medium for the given grid
pub fn create_test_medium(grid: &Grid) -> HomogeneousMedium {
    HomogeneousMedium::new(1000.0, 1500.0, 0.1, 1.0, grid)
}

/// Create a test field with specified dimensions
pub fn create_test_field(nx: usize, ny: usize, nz: usize) -> Array3<f64> {
    let mut field = Array3::zeros((nx, ny, nz));
    // Add a simple pattern for testing
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let x = i as f64 / nx as f64;
                let y = j as f64 / ny as f64;
                let z = k as f64 / nz as f64;
                field[[i, j, k]] = (x * x + y * y + z * z).sqrt();
            }
        }
    }
    field
}

/// Create a default test field (32x32x32)
pub fn create_default_test_field() -> Array3<f64> {
    create_test_field(32, 32, 32)
}
