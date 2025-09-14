//! Grid structure and basic operations
//!
//! This module defines the core grid structure for spatial discretization.

use crate::error::grid::GridError;
use log::debug;
use ndarray::Array3;
use std::sync::OnceLock;

/// Epsilon for floating point comparisons of grid spacing
const GRID_SPACING_EQUALITY_EPSILON: f64 = 1e-10;

/// Spatial bounds for a region
#[derive(Debug, Clone, Copy)]
pub struct Bounds {
    /// Minimum coordinates [x, y, z]
    pub min: [f64; 3],
    /// Maximum coordinates [x, y, z]
    pub max: [f64; 3],
}

impl Bounds {
    /// Create new bounds
    #[must_use]
    pub fn new(min: [f64; 3], max: [f64; 3]) -> Self {
        Self { min, max }
    }

    /// Get the center point
    #[must_use]
    pub fn center(&self) -> [f64; 3] {
        [
            (self.min[0] + self.max[0]) / 2.0,
            (self.min[1] + self.max[1]) / 2.0,
            (self.min[2] + self.max[2]) / 2.0,
        ]
    }
}

/// Dimension selector for coordinate generation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dimension {
    X,
    Y,
    Z,
}

/// Defines a 3D Cartesian grid for the simulation domain
#[derive(Debug, Clone)]
pub struct Grid {
    /// Number of points in x-direction
    pub nx: usize,
    /// Number of points in y-direction
    pub ny: usize,
    /// Number of points in z-direction
    pub nz: usize,
    /// Spacing in x-direction (meters)
    pub dx: f64,
    /// Spacing in y-direction (meters)
    pub dy: f64,
    /// Spacing in z-direction (meters)
    pub dz: f64,
    /// Cache for `k_squared` computation
    pub(crate) k_squared_cache: OnceLock<Array3<f64>>,
}

impl Default for Grid {
    /// Creates a default 32x32x32 grid with 1mm spacing
    fn default() -> Self {
        // Direct construction since we know the values are valid
        Self {
            nx: 32,
            ny: 32,
            nz: 32,
            dx: 1e-3,
            dy: 1e-3,
            dz: 1e-3,
            k_squared_cache: OnceLock::new(),
        }
    }
}

impl Grid {
    /// Creates a new grid with specified dimensions and spacing.
    /// Returns a `GridError` if parameters are invalid.
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> Result<Self, GridError> {
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(GridError::ZeroDimension { nx, ny, nz });
        }
        if dx <= 0.0 || dy <= 0.0 || dz <= 0.0 {
            return Err(GridError::NonPositiveSpacing { dx, dy, dz });
        }

        let grid = Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            k_squared_cache: OnceLock::new(),
        };

        debug!(
            "Created grid: {}x{}x{} points, spacing: ({:.3e}, {:.3e}, {:.3e}) m",
            nx, ny, nz, dx, dy, dz
        );

        Ok(grid)
    }

    /// Creates a new grid with the same spacing in all directions
    pub fn uniform(n: usize, spacing: f64) -> Result<Self, GridError> {
        Self::new(n, n, n, spacing, spacing, spacing)
    }

    /// Get the total number of grid points
    #[inline]
    pub fn size(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// Get dimensions as a tuple
    #[inline]
    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }

    /// Get spacing as a tuple
    #[inline]
    pub fn spacing(&self) -> (f64, f64, f64) {
        (self.dx, self.dy, self.dz)
    }

    /// Check if the grid has uniform spacing
    #[inline]
    pub fn is_uniform(&self) -> bool {
        (self.dx - self.dy).abs() < GRID_SPACING_EQUALITY_EPSILON
            && (self.dy - self.dz).abs() < GRID_SPACING_EQUALITY_EPSILON
    }

    /// Convert grid indices to physical coordinates
    #[inline]
    pub fn indices_to_coordinates(&self, i: usize, j: usize, k: usize) -> (f64, f64, f64) {
        (i as f64 * self.dx, j as f64 * self.dy, k as f64 * self.dz)
    }

    /// Get an iterator over the coordinates for a specific dimension.
    /// This returns a statically-dispatched iterator with no heap allocation.
    pub fn coordinates(&self, dim: Dimension) -> impl Iterator<Item = f64> + '_ {
        let (count, spacing) = match dim {
            Dimension::X => (self.nx, self.dx),
            Dimension::Y => (self.ny, self.dy),
            Dimension::Z => (self.nz, self.dz),
        };
        (0..count).map(move |i| i as f64 * spacing)
    }

    /// Convert position to grid indices
    #[inline]
    pub fn position_to_indices(&self, x: f64, y: f64, z: f64) -> Option<(usize, usize, usize)> {
        let max_x = self.nx as f64 * self.dx;
        let max_y = self.ny as f64 * self.dy;
        let max_z = self.nz as f64 * self.dz;

        if x < 0.0 || y < 0.0 || z < 0.0 || x > max_x || y > max_y || z > max_z {
            return None;
        }

        // Using floor is safer and more predictable than rounding
        let i = (x / self.dx).floor() as usize;
        let j = (y / self.dy).floor() as usize;
        let k = (z / self.dz).floor() as usize;

        // Clamp to ensure the index is within bounds, preventing off-by-one due to floating point issues
        Some((i.min(self.nx - 1), j.min(self.ny - 1), k.min(self.nz - 1)))
    }

    /// Create a zero-initialized field with grid dimensions
    #[inline]
    pub fn create_field(&self) -> ndarray::Array3<f64> {
        ndarray::Array3::zeros((self.nx, self.ny, self.nz))
    }
}
