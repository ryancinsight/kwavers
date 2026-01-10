//! Grid structure and basic operations
//!
//! This module defines the core grid structure for spatial discretization.

use super::error::GridError;
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
    /// Origin of the grid [x0, y0, z0]
    pub origin: [f64; 3],
    /// Dimensionality of the grid (1, 2, or 3)
    pub dimensionality: usize,
    /// Maximum wavenumber supported by the grid
    pub k_max: f64,
    /// Cache for `k_squared` computation
    pub(crate) k_squared_cache: OnceLock<Array3<f64>>,
}

impl Default for Grid {
    /// Creates a default 32x32x32 grid with 1mm spacing
    fn default() -> Self {
        // Direct construction since we know the values are valid
        let dx = 1e-3;
        let dy = 1e-3;
        let dz = 1e-3;
        let k_max = std::f64::consts::PI / dx;

        Self {
            nx: 32,
            ny: 32,
            nz: 32,
            dx,
            dy,
            dz,
            origin: [0.0, 0.0, 0.0],
            dimensionality: 3,
            k_max,
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

        let mut dimensionality = 0;
        if nx > 1 {
            dimensionality += 1;
        }
        if ny > 1 {
            dimensionality += 1;
        }
        if nz > 1 {
            dimensionality += 1;
        }
        if dimensionality == 0 {
            dimensionality = 1;
        } // Scalar point

        let min_dx = if nx > 1 { dx } else { f64::MAX }
            .min(if ny > 1 { dy } else { f64::MAX })
            .min(if nz > 1 { dz } else { f64::MAX });

        // Handle cases where all dimensions are 1
        let min_dx = if min_dx == f64::MAX { dx } else { min_dx };
        let k_max = std::f64::consts::PI / min_dx;

        let grid = Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            origin: [0.0, 0.0, 0.0],
            dimensionality,
            k_max,
            k_squared_cache: OnceLock::new(),
        };

        debug!(
            "Created grid: {}x{}x{} points, spacing: ({:.3e}, {:.3e}, {:.3e}) m",
            nx, ny, nz, dx, dy, dz
        );

        Ok(grid)
    }

    pub(crate) fn k_squared(&self) -> &Array3<f64> {
        self.k_squared_cache.get_or_init(|| {
            let mut k2 = Array3::<f64>::zeros((self.nx, self.ny, self.nz));
            for i in 0..self.nx {
                for j in 0..self.ny {
                    for k in 0..self.nz {
                        let kx = std::f64::consts::PI * (i as f64) / (self.nx as f64 * self.dx);
                        let ky = std::f64::consts::PI * (j as f64) / (self.ny as f64 * self.dy);
                        let kz = std::f64::consts::PI * (k as f64) / (self.nz as f64 * self.dz);
                        k2[[i, j, k]] = kx * kx + ky * ky + kz * kz;
                    }
                }
            }
            k2
        })
    }
}

    /// Creates a new grid with the same spacing in all directions
    pub fn uniform(n: usize, spacing: f64) -> Result<Self, GridError> {
        Self::new(n, n, n, spacing, spacing, spacing)
    }

    /// Create a grid optimized for a given frequency
    /// Domain-specific factory method following Information Expert
    pub fn create_for_frequency(
        frequency: f64,
        sound_speed: f64,
        points_per_wavelength: usize,
    ) -> Result<Self, GridError> {
        let wavelength = sound_speed / frequency;
        let spacing = wavelength / points_per_wavelength as f64;

        // Calculate reasonable grid size based on wavelength (approx 10 wavelengths)
        let size = ((wavelength * 10.0) / spacing) as usize;
        let size = size.max(16); // Minimum 16 points per dimension

        Self::new(size, size, size, spacing, spacing, spacing)
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

    /// Get total number of grid points (alias for size)
    #[inline]
    pub fn total_grid_points(&self) -> usize {
        self.size()
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
