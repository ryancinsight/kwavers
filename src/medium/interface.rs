//! Interface detection utilities for medium boundaries
//!
//! This module provides utilities for detecting interfaces and boundaries
//! between different media based on property discontinuities.

use crate::grid::Grid;
use crate::medium::core::{ArrayAccess, CoreMedium};
use ndarray::Array3;

/// Interface point information
#[derive(Debug, Clone)]
pub struct InterfacePoint {
    /// Position in world coordinates (x, y, z)
    pub position: (f64, f64, f64),
    /// Grid indices (i, j, k)
    pub indices: (usize, usize, usize),
    /// Density jump across the interface
    pub density_jump: f64,
    /// Normal vector at the interface
    pub normal: [f64; 3],
}

/// Efficiently find interfaces in a medium based on density jumps
///
/// # Arguments
/// * `medium` - The medium to analyze
/// * `grid` - The computational grid
/// * `threshold` - Relative density jump threshold (e.g., 0.1 for 10% change)
///
/// # Returns
/// Vector of interface points where density discontinuities exceed the threshold
pub fn find_interfaces<M: CoreMedium + ArrayAccess + ?Sized>(
    medium: &M,
    grid: &Grid,
    threshold: f64,
) -> Vec<InterfacePoint> {
    find_interfaces_from_array(&medium.density_array(), grid, threshold)
}

/// Find interfaces using point-wise access (less efficient fallback)
pub fn find_interfaces_pointwise<M: CoreMedium + ?Sized>(
    medium: &M,
    grid: &Grid,
    threshold: f64,
) -> Vec<InterfacePoint> {
    let mut interfaces = Vec::new();

    for i in 1..(grid.nx - 1) {
        for j in 1..(grid.ny - 1) {
            for k in 1..(grid.nz - 1) {
                let (x, y, z) = grid.coordinates(i, j, k);
                let center_density = medium.density(x, y, z, grid);

                // Check one neighbor for efficiency
                let (nx, ny, nz) = grid.coordinates(i + 1, j, k);
                let neighbor_density = medium.density(nx, ny, nz, grid);

                if ((neighbor_density - center_density).abs() / center_density) > threshold {
                    // Simplified normal calculation
                    let normal = [1.0, 0.0, 0.0]; // Assume x-direction for simplicity

                    interfaces.push(InterfacePoint {
                        position: (x, y, z),
                        indices: (i, j, k),
                        density_jump: neighbor_density - center_density,
                        normal,
                    });
                }
            }
        }
    }

    interfaces
}

/// Efficient array-based interface detection
fn find_interfaces_from_array(
    density_array: &Array3<f64>,
    grid: &Grid,
    threshold: f64,
) -> Vec<InterfacePoint> {
    let mut interfaces = Vec::new();

    // Check interior points only (avoid boundaries)
    for i in 1..(grid.nx - 1) {
        for j in 1..(grid.ny - 1) {
            for k in 1..(grid.nz - 1) {
                let center_density = density_array[[i, j, k]];

                // Check neighbors for density jump
                let neighbors = [
                    (i + 1, j, k),
                    (i - 1, j, k),
                    (i, j + 1, k),
                    (i, j - 1, k),
                    (i, j, k + 1),
                    (i, j, k - 1),
                ];

                for &(ni, nj, nk) in &neighbors {
                    let neighbor_density = density_array[[ni, nj, nk]];
                    let relative_jump = (neighbor_density - center_density).abs() / center_density;

                    if relative_jump > threshold {
                        // Calculate normal using central differences
                        let dx = (density_array[[i + 1, j, k]] - density_array[[i - 1, j, k]])
                            / (2.0 * grid.dx);
                        let dy = (density_array[[i, j + 1, k]] - density_array[[i, j - 1, k]])
                            / (2.0 * grid.dy);
                        let dz = (density_array[[i, j, k + 1]] - density_array[[i, j, k - 1]])
                            / (2.0 * grid.dz);

                        let norm = (dx * dx + dy * dy + dz * dz).sqrt();
                        let normal = if norm > 1e-10 {
                            [dx / norm, dy / norm, dz / norm]
                        } else {
                            [0.0, 0.0, 0.0]
                        };

                        interfaces.push(InterfacePoint {
                            position: grid.coordinates(i, j, k),
                            indices: (i, j, k),
                            density_jump: neighbor_density - center_density,
                            normal,
                        });

                        break; // Only record once per grid point
                    }
                }
            }
        }
    }

    interfaces
}
