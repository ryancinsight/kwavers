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
use aequitas::systems::si::quantities::{Length, Time, Velocity, Volume};
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
pub use geometry::{
    GeometricDomain, GeometryDimension, GeometryError, PointLocation, RectangularDomain,
    SphericalDomain,
};

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
    /// Return the minimum grid spacing as an SI length.
    #[inline]
    #[must_use]
    pub fn min_spacing(&self) -> Length<f64> {
        Length::from_base(self.dx.min(self.dy).min(self.dz))
    }

    /// Return the maximum grid spacing as an SI length.
    #[inline]
    #[must_use]
    pub fn max_spacing(&self) -> Length<f64> {
        Length::from_base(self.dx.max(self.dy).max(self.dz))
    }

    /// Return the physical domain dimensions as SI lengths.
    #[inline]
    pub fn physical_size(&self) -> (Length<f64>, Length<f64>, Length<f64>) {
        (
            Length::from_base(self.nx as f64 * self.dx),
            Length::from_base(self.ny as f64 * self.dy),
            Length::from_base(self.nz as f64 * self.dz),
        )
    }

    /// Return the physical grid volume in SI units.
    #[inline]
    #[must_use]
    pub fn volume(&self) -> Volume<f64> {
        let (lx, ly, lz) = self.physical_size();
        lx * ly * lz
    }

    /// Return the volume of one grid cell in SI units.
    #[inline]
    #[must_use]
    pub fn cell_volume(&self) -> Volume<f64> {
        Length::from_base(self.dx) * Length::from_base(self.dy) * Length::from_base(self.dz)
    }

    /// Check if a point is inside the grid
    #[inline]
    pub fn contains_point(&self, x: f64, y: f64, z: f64) -> bool {
        let (lx, ly, lz) = self.physical_size();
        let (lx, ly, lz) = (lx.into_base(), ly.into_base(), lz.into_base());
        x >= 0.0 && x <= lx && y >= 0.0 && y <= ly && z >= 0.0 && z <= lz
    }

    /// Get the bounds of the grid
    #[inline]
    pub fn bounds(&self) -> Bounds {
        let (lx, ly, lz) = self.physical_size();
        Bounds::new(
            [0.0, 0.0, 0.0],
            [lx.into_base(), ly.into_base(), lz.into_base()],
        )
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

    /// Calculate the CFL timestep for a given sound speed.
    ///
    /// Uses the FDTD stability condition with a safety factor. The input and
    /// output carry their SI dimensions at this public boundary; the scalar
    /// stability kernel remains dimensionless internally.
    #[inline]
    #[must_use]
    pub fn cfl_timestep(&self, max_sound_speed: Velocity<f64>) -> Time<f64> {
        Time::from_base(stability::StabilityCalculator::cfl_timestep_fdtd(
            self,
            max_sound_speed.into_base(),
        ))
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

#[cfg(test)]
mod tests {
    use super::Grid;
    use aequitas::systems::si::quantities::Velocity;

    #[test]
    fn derived_metrics_preserve_si_values() {
        let grid = Grid::new(4, 5, 6, 0.2, 0.3, 0.4).expect("valid grid dimensions");

        assert_eq!(grid.min_spacing().into_base(), 0.2);
        assert_eq!(grid.max_spacing().into_base(), 0.4);

        let (length_x, length_y, length_z) = grid.physical_size();
        assert_eq!(length_x.into_base(), 0.8);
        assert_eq!(length_y.into_base(), 1.5);
        assert_eq!(length_z.into_base(), 2.4);

        let expected_cell_volume = 0.2 * 0.3 * 0.4;
        let expected_volume = 0.8 * 1.5 * 2.4;
        assert_eq!(grid.cell_volume().into_base(), expected_cell_volume);
        assert_eq!(grid.volume().into_base(), expected_volume);

        let timestep = grid.cfl_timestep(Velocity::from_base(1500.0));
        assert!(timestep.into_base().is_sign_positive());
        assert!(timestep.into_base().is_finite());
    }
}
