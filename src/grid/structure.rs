//! Grid structure and basic operations
//!
//! This module defines the core grid structure for spatial discretization.

use crate::error::{KwaversError, KwaversResult};
use log::debug;
use ndarray::Array3;
use std::sync::OnceLock;

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
    pub fn new(min: [f64; 3], max: [f64; 3]) -> Self {
        Self { min, max }
    }

    /// Get the center point
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
    /// Cache for k_squared computation
    pub(crate) k_squared_cache: OnceLock<Array3<f64>>,
}

impl Default for Grid {
    /// Creates a default 32x32x32 grid with 1mm spacing
    fn default() -> Self {
        Self::new(32, 32, 32, 1e-3, 1e-3, 1e-3)
    }
}

impl Grid {
    /// Creates a new grid with specified dimensions and spacing
    ///
    /// # Panics
    /// Panics if dimensions or spacing are not positive
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64) -> Self {
        Self::create(nx, ny, nz, dx, dy, dz).expect("Invalid grid parameters")
    }

    /// Creates a new grid, returning an error if invalid
    pub fn create(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> KwaversResult<Self> {
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(KwaversError::InvalidInput(format!(
                "Grid dimensions must be positive, got nx={}, ny={}, nz={}",
                nx, ny, nz
            )));
        }
        if dx <= 0.0 || dy <= 0.0 || dz <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "Grid spacing must be positive, got dx={}, dy={}, dz={}",
                dx, dy, dz
            )));
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
            "Grid created: {}x{}x{}, dx = {}, dy = {}, dz = {}",
            nx, ny, nz, dx, dy, dz
        );

        if (dx - dy).abs() > 1e-10 || (dy - dz).abs() > 1e-10 {
            debug!("Warning: Non-uniform grid spacing may affect k-space accuracy");
        }

        Ok(grid)
    }

    /// Total number of grid points
    #[inline]
    pub fn total_points(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// Physical dimensions of the grid (meters)
    #[inline]
    pub fn physical_dimensions(&self) -> (f64, f64, f64) {
        (
            self.nx as f64 * self.dx,
            self.ny as f64 * self.dy,
            self.nz as f64 * self.dz,
        )
    }

    /// Check if grid has uniform spacing
    #[inline]
    pub fn is_uniform(&self) -> bool {
        (self.dx - self.dy).abs() < 1e-10 && (self.dy - self.dz).abs() < 1e-10
    }

    /// Minimum grid spacing
    #[inline]
    pub fn min_spacing(&self) -> f64 {
        self.dx.min(self.dy).min(self.dz)
    }

    /// Maximum grid spacing
    #[inline]
    pub fn max_spacing(&self) -> f64 {
        self.dx.max(self.dy).max(self.dz)
    }

    /// Get the spatial bounds of the grid
    pub fn bounds(&self) -> Bounds {
        Bounds::new(
            [0.0, 0.0, 0.0],
            [
                self.nx as f64 * self.dx,
                self.ny as f64 * self.dy,
                self.nz as f64 * self.dz,
            ],
        )
    }
}
