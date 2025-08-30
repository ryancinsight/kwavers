//! Coordinate generation and position conversions
//!
//! This module handles physical coordinate generation and position-to-index conversions.

use crate::grid::structure::{Dimension, Grid};
use ndarray::{Array1, Array3};

/// Coordinate system operations
#[derive(Debug, Debug)]
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
        Array1::from_shape_fn(grid.nx, |i| i as f64 * grid.dx)
    }

    /// Generate y-coordinate vector
    pub fn generate_y_vector(grid: &Grid) -> Array1<f64> {
        Array1::from_shape_fn(grid.ny, |j| j as f64 * grid.dy)
    }

    /// Generate z-coordinate vector
    pub fn generate_z_vector(grid: &Grid) -> Array1<f64> {
        Array1::from_shape_fn(grid.nz, |k| k as f64 * grid.dz)
    }

    /// Generate 3D coordinate arrays
    pub fn generate_coordinate_arrays(grid: &Grid) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
        let mut x_coords = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut y_coords = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut z_coords = Array3::zeros((grid.nx, grid.ny, grid.nz));

        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    x_coords[[i, j, k]] = i as f64 * grid.dx;
                    y_coords[[i, j, k]] = j as f64 * grid.dy;
                    z_coords[[i, j, k]] = k as f64 * grid.dz;
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
        if x < 0.0 || y < 0.0 || z < 0.0 {
            return None;
        }

        let i = (x / grid.dx).floor() as usize;
        let j = (y / grid.dy).floor() as usize;
        let k = (z / grid.dz).floor() as usize;

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
        if x < 0.0 || y < 0.0 || z < 0.0 {
            return None;
        }

        let i = (x / grid.dx).round() as usize;
        let j = (y / grid.dy).round() as usize;
        let k = (z / grid.dz).round() as usize;

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
                (i as f64 + 0.5) * grid.dx,
                (j as f64 + 0.5) * grid.dy,
                (k as f64 + 0.5) * grid.dz,
            ))
        }
    }

    /// Check if position is within grid bounds
    pub fn is_position_in_bounds(grid: &Grid, x: f64, y: f64, z: f64) -> bool {
        x >= 0.0
            && y >= 0.0
            && z >= 0.0
            && x < grid.nx as f64 * grid.dx
            && y < grid.ny as f64 * grid.dy
            && z < grid.nz as f64 * grid.dz
    }
}
