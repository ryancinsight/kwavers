//! Spatial discretization for kwavers.
//!
//! This crate provides the core grid structures and utilities for defining
//! computational domains and spatial discretization (Cartesian/cylindrical
//! grids, coordinates, topology, operators, k-space FFT helpers), plus the
//! geometric-domain primitives (rectangular/spherical regions, point
//! classification) previously co-located in `kwavers-domain`.

pub mod adapter;
pub mod config;
pub mod coordinates;
pub mod error;
pub mod geometry;
// field_ops moved to domain/field/operations.rs
use kwavers_math::fft::kspace;
pub mod operators;
pub mod simple_config;
pub mod stability;
pub mod structure;
pub mod topology;
pub mod validation;
use kwavers_math::fft::utils as fft_utils;

// ============================================================================
// EXPLICIT RE-EXPORTS (Grid API)
// ============================================================================

/// Core grid structure and dimensional types
pub use structure::{Bounds, Grid, GridDimension};

/// Grid configuration types
pub use config::{DomainGridParameters, GridType};
pub use simple_config::GridConfig;

/// Coordinate systems
pub use coordinates::CoordinateSystem;

/// Topology definitions
pub use topology::{CartesianTopology, CylindricalTopology, GridTopology, TopologyDimension};

/// Grid adapter and extension traits
pub use adapter::{GridAdapter, GridTopologyExt};

/// Grid validation
pub use validation::GridValidator;

/// FFT utilities for k-space operations
pub use fft_utils::{fft_shift_2d, get_optimal_fft_size, ifft_shift_2d, is_optimal_fft_size};
pub use kspace::KSpaceCalculator;

/// Geometric-domain primitives
pub use geometry::{GeometricDomain, PointLocation, RectangularDomain, SphericalDomain};

pub type Grid3D = Grid;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridDimensions {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
}

impl GridDimensions {
    /// New.
    /// # Panics
    /// - Panics if assertion fails: `Grid dimensions must be non-zero`.
    ///
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64) -> Self {
        assert!(
            nx > 0 && ny > 0 && nz > 0,
            "Grid dimensions must be non-zero"
        );
        assert!(
            dx.is_finite() && dy.is_finite() && dz.is_finite() && dx > 0.0 && dy > 0.0 && dz > 0.0,
            "Grid spacing must be finite and positive"
        );
        Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
        }
    }

    #[must_use]
    pub fn from_grid(grid: &Grid) -> Self {
        Self {
            nx: grid.nx,
            ny: grid.ny,
            nz: grid.nz,
            dx: grid.dx,
            dy: grid.dy,
            dz: grid.dz,
        }
    }
}

// Extension methods for Grid
impl Grid {
    /// Get minimum spacing
    #[inline]
    #[must_use]
    pub fn min_spacing(&self) -> f64 {
        self.dx.min(self.dy).min(self.dz)
    }

    /// Get maximum spacing
    #[inline]
    #[must_use]
    pub fn max_spacing(&self) -> f64 {
        self.dx.max(self.dy).max(self.dz)
    }

    /// Get physical dimensions of the domain
    #[inline]
    pub fn physical_size(&self) -> (f64, f64, f64) {
        (
            self.nx as f64 * self.dx,
            self.ny as f64 * self.dy,
            self.nz as f64 * self.dz,
        )
    }

    /// Get grid volume
    #[inline]
    #[must_use]
    pub fn volume(&self) -> f64 {
        let (lx, ly, lz) = self.physical_size();
        lx * ly * lz
    }

    /// Get cell volume
    #[inline]
    #[must_use]
    pub fn cell_volume(&self) -> f64 {
        self.dx * self.dy * self.dz
    }

    /// Check if a point is inside the grid
    #[inline]
    pub fn contains_point(&self, x: f64, y: f64, z: f64) -> bool {
        let (lx, ly, lz) = self.physical_size();
        x >= 0.0 && x <= lx && y >= 0.0 && y <= ly && z >= 0.0 && z <= lz
    }

    /// Get the bounds of the grid
    #[inline]
    pub fn bounds(&self) -> Bounds {
        let (lx, ly, lz) = self.physical_size();
        Bounds::new([0.0, 0.0, 0.0], [lx, ly, lz])
    }

    /// Convert coordinates to indices (nearest neighbor)
    #[inline]
    pub fn coordinates_to_indices(&self, x: f64, y: f64, z: f64) -> Option<(usize, usize, usize)> {
        if !self.contains_point(x, y, z) {
            return None;
        }
        Some((
            (x / self.dx) as usize,
            (y / self.dy) as usize,
            (z / self.dz) as usize,
        ))
    }

    /// Compute the x-axis wave-number vector.
    #[inline]
    pub fn compute_kx(&self) -> leto::Array1<f64> {
        KSpaceCalculator::generate_k_vector(self.nx, self.dx)
    }

    /// Compute the y-axis wave-number vector.
    #[inline]
    pub fn compute_ky(&self) -> leto::Array1<f64> {
        KSpaceCalculator::generate_k_vector(self.ny, self.dy)
    }

    /// Compute the z-axis wave-number vector.
    #[inline]
    pub fn compute_kz(&self) -> leto::Array1<f64> {
        KSpaceCalculator::generate_k_vector(self.nz, self.dz)
    }

    /// Calculate CFL timestep for given sound speed
    ///
    /// Uses FDTD stability condition with safety factor
    #[inline]
    #[must_use]
    pub fn cfl_timestep(&self, max_sound_speed: f64) -> f64 {
        stability::StabilityCalculator::cfl_timestep_fdtd(self, max_sound_speed)
    }

    /// Get x coordinates (compatibility)
    #[inline]
    pub fn x_coordinates(&self) -> leto::Array1<f64> {
        CoordinateSystem::generate_x_vector(self)
    }

    /// Get y coordinates (compatibility)
    #[inline]
    pub fn y_coordinates(&self) -> leto::Array1<f64> {
        CoordinateSystem::generate_y_vector(self)
    }

    /// Get z coordinates (compatibility)
    #[inline]
    pub fn z_coordinates(&self) -> leto::Array1<f64> {
        CoordinateSystem::generate_z_vector(self)
    }

    /// Get maximum wavenumber supported by grid
    #[inline]
    pub fn k_max(&self) -> f64 {
        self.k_max
    }
}
