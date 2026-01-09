//! Coordinate generation and position conversions
//!
//! This module handles physical coordinate generation and position-to-index conversions.

use crate::domain::grid::structure::{Dimension, Grid};
use ndarray::{Array1, Array3};

/// Coordinate system operations
#[derive(Debug)]
pub struct CoordinateSystem;

impl CoordinateSystem {
    /// Generate coordinate vector for a given dimension
    pub fn generate_coordinate_vector(grid: &Grid, dim: Dimension) -> Array1<f64> {
        match dim {
            Dimension::X => Self::generate_x_vector(grid),
            Dimension::Y => Self::generate_y_vector(grid),
            Dimension::Z => Self::generate_z_vector(grid),
        }
    }

    /// Generate x-coordinate vector
    pub fn generate_x_vector(grid: &Grid) -> Array1<f64> {
        Array1::from_shape_fn(grid.nx, |i| grid.origin[0] + i as f64 * grid.dx)
    }

    /// Generate y-coordinate vector
    pub fn generate_y_vector(grid: &Grid) -> Array1<f64> {
        Array1::from_shape_fn(grid.ny, |j| grid.origin[1] + j as f64 * grid.dy)
    }

    /// Generate z-coordinate vector
    pub fn generate_z_vector(grid: &Grid) -> Array1<f64> {
        Array1::from_shape_fn(grid.nz, |k| grid.origin[2] + k as f64 * grid.dz)
    }

    /// Generate 3D coordinate arrays
    pub fn generate_coordinate_arrays(grid: &Grid) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
        let mut x_coords = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut y_coords = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut z_coords = Array3::zeros((grid.nx, grid.ny, grid.nz));

        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    x_coords[[i, j, k]] = grid.origin[0] + i as f64 * grid.dx;
                    y_coords[[i, j, k]] = grid.origin[1] + j as f64 * grid.dy;
                    z_coords[[i, j, k]] = grid.origin[2] + k as f64 * grid.dz;
                }
            }
        }

        (x_coords, y_coords, z_coords)
    }

    /// Convert physical position to grid indices
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
                grid.origin[0] + (i as f64 + 0.5) * grid.dx,
                grid.origin[1] + (j as f64 + 0.5) * grid.dy,
                grid.origin[2] + (k as f64 + 0.5) * grid.dz,
            ))
        }
    }

    /// Check if position is within grid bounds
    pub fn is_position_in_bounds(grid: &Grid, x: f64, y: f64, z: f64) -> bool {
        x >= grid.origin[0]
            && y >= grid.origin[1]
            && z >= grid.origin[2]
            && x < grid.origin[0] + grid.nx as f64 * grid.dx
            && y < grid.origin[1] + grid.ny as f64 * grid.dy
            && z < grid.origin[2] + grid.nz as f64 * grid.dz
    }

    /// Generate centered coordinate vector for a given dimension
    pub fn generate_centered_coordinate_vector(grid: &Grid, dim: Dimension) -> Array1<f64> {
        let (n, d, o) = match dim {
            Dimension::X => (grid.nx, grid.dx, grid.origin[0]),
            Dimension::Y => (grid.ny, grid.dy, grid.origin[1]),
            Dimension::Z => (grid.nz, grid.dz, grid.origin[2]),
        };
        let length = n as f64 * d;
        Array1::from_shape_fn(n, |i| o + (i as f64 * d) - length / 2.0 + d / 2.0)
    }

    /// Generate 3D centered coordinate arrays
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
                    x_coords[[i, j, k]] = origin[0] + i as f64 * dx - lx / 2.0 + dx / 2.0;
                    y_coords[[i, j, k]] = origin[1] + j as f64 * dy - ly / 2.0 + dy / 2.0;
                    z_coords[[i, j, k]] = origin[2] + k as f64 * dz - lz / 2.0 + dz / 2.0;
                }
            }
        }

        (x_coords, y_coords, z_coords)
    }
}
