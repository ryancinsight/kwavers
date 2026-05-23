//! Coordinate generation and position conversions
//!
//! This module handles physical coordinate generation and position-to-index conversions.

use crate::domain::grid::structure::{Grid, GridDimension};
use ndarray::{Array1, Array3};

/// Coordinate system operations
#[derive(Debug)]
pub struct CoordinateSystem;

impl CoordinateSystem {
    /// Generate coordinate vector for a given dimension
    #[must_use]
    pub fn generate_coordinate_vector(grid: &Grid, dim: GridDimension) -> Array1<f64> {
        match dim {
            GridDimension::X => Self::generate_x_vector(grid),
            GridDimension::Y => Self::generate_y_vector(grid),
            GridDimension::Z => Self::generate_z_vector(grid),
        }
    }

    /// Generate x-coordinate vector
    #[must_use]
    pub fn generate_x_vector(grid: &Grid) -> Array1<f64> {
        Array1::from_shape_fn(grid.nx, |i| (i as f64).mul_add(grid.dx, grid.origin[0]))
    }

    /// Generate y-coordinate vector
    #[must_use]
    pub fn generate_y_vector(grid: &Grid) -> Array1<f64> {
        Array1::from_shape_fn(grid.ny, |j| (j as f64).mul_add(grid.dy, grid.origin[1]))
    }

    /// Generate z-coordinate vector
    #[must_use]
    pub fn generate_z_vector(grid: &Grid) -> Array1<f64> {
        Array1::from_shape_fn(grid.nz, |k| (k as f64).mul_add(grid.dz, grid.origin[2]))
    }

    /// Generate 3D coordinate arrays
    #[must_use]
    pub fn generate_coordinate_arrays(grid: &Grid) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
        let mut x_coords = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut y_coords = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut z_coords = Array3::zeros((grid.nx, grid.ny, grid.nz));

        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    x_coords[[i, j, k]] = (i as f64).mul_add(grid.dx, grid.origin[0]);
                    y_coords[[i, j, k]] = (j as f64).mul_add(grid.dy, grid.origin[1]);
                    z_coords[[i, j, k]] = (k as f64).mul_add(grid.dz, grid.origin[2]);
                }
            }
        }

        (x_coords, y_coords, z_coords)
    }

    /// Convert physical position to grid indices
    #[must_use]
    pub fn position_to_indices(
        grid: &Grid,
        x: f64,
        y: f64,
        z: f64,
    ) -> Option<(usize, usize, usize)> {
        // Check bounds
        if x < grid.origin[0] || y < grid.origin[1] || z < grid.origin[2] {
            return None;
        }

        let i = ((x - grid.origin[0]) / grid.dx).floor() as usize;
        let j = ((y - grid.origin[1]) / grid.dy).floor() as usize;
        let k = ((z - grid.origin[2]) / grid.dz).floor() as usize;

        if i >= grid.nx || j >= grid.ny || k >= grid.nz {
            None
        } else {
            Some((i, j, k))
        }
    }

    /// Convert physical position to nearest grid indices
    #[must_use]
    pub fn position_to_nearest_indices(
        grid: &Grid,
        x: f64,
        y: f64,
        z: f64,
    ) -> Option<(usize, usize, usize)> {
        if x < grid.origin[0] || y < grid.origin[1] || z < grid.origin[2] {
            return None;
        }

        let i = ((x - grid.origin[0]) / grid.dx).round() as usize;
        let j = ((y - grid.origin[1]) / grid.dy).round() as usize;
        let k = ((z - grid.origin[2]) / grid.dz).round() as usize;

        if i >= grid.nx || j >= grid.ny || k >= grid.nz {
            None
        } else {
            Some((i, j, k))
        }
    }

    /// Convert grid indices to physical position (center of cell)
    #[must_use]
    pub fn indices_to_position(
        grid: &Grid,
        i: usize,
        j: usize,
        k: usize,
    ) -> Option<(f64, f64, f64)> {
        if i >= grid.nx || j >= grid.ny || k >= grid.nz {
            None
        } else {
            Some((
                (i as f64 + 0.5).mul_add(grid.dx, grid.origin[0]),
                (j as f64 + 0.5).mul_add(grid.dy, grid.origin[1]),
                (k as f64 + 0.5).mul_add(grid.dz, grid.origin[2]),
            ))
        }
    }

    /// Check if position is within grid bounds
    #[must_use]
    pub fn is_position_in_bounds(grid: &Grid, x: f64, y: f64, z: f64) -> bool {
        x >= grid.origin[0]
            && y >= grid.origin[1]
            && z >= grid.origin[2]
            && x < (grid.nx as f64).mul_add(grid.dx, grid.origin[0])
            && y < (grid.ny as f64).mul_add(grid.dy, grid.origin[1])
            && z < (grid.nz as f64).mul_add(grid.dz, grid.origin[2])
    }

    /// Generate centered coordinate vector for a given dimension
    #[must_use]
    pub fn generate_centered_coordinate_vector(grid: &Grid, dim: GridDimension) -> Array1<f64> {
        let (n, d, o) = match dim {
            GridDimension::X => (grid.nx, grid.dx, grid.origin[0]),
            GridDimension::Y => (grid.ny, grid.dy, grid.origin[1]),
            GridDimension::Z => (grid.nz, grid.dz, grid.origin[2]),
        };
        let length = n as f64 * d;
        Array1::from_shape_fn(n, |i| (i as f64).mul_add(d, o) - length / 2.0 + d / 2.0)
    }

    /// Generate 3D centered coordinate arrays
    #[must_use]
    pub fn generate_centered_coordinate_arrays(
        grid: &Grid,
    ) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
        let (nx, ny, nz) = grid.dimensions();
        let (dx, dy, dz) = grid.spacing();
        let origin = grid.origin;

        let lx = nx as f64 * dx;
        let ly = ny as f64 * dy;
        let lz = nz as f64 * dz;

        let mut x_coords = Array3::zeros((nx, ny, nz));
        let mut y_coords = Array3::zeros((nx, ny, nz));
        let mut z_coords = Array3::zeros((nx, ny, nz));

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    x_coords[[i, j, k]] = (i as f64).mul_add(dx, origin[0]) - lx / 2.0 + dx / 2.0;
                    y_coords[[i, j, k]] = (j as f64).mul_add(dy, origin[1]) - ly / 2.0 + dy / 2.0;
                    z_coords[[i, j, k]] = (k as f64).mul_add(dz, origin[2]) - lz / 2.0 + dz / 2.0;
                }
            }
        }

        (x_coords, y_coords, z_coords)
    }
}
