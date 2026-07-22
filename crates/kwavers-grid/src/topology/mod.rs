//! Grid topology trait and implementations for different coordinate systems
//!
//! Defines the `GridTopology` trait that abstracts over coordinate systems
//! (Cartesian, cylindrical, spherical, etc.) and provides a unified interface
//! for spatial discretization.
//!
//! # Invariants
//!
//! - Grid spacing must be positive and finite
//! - Dimensions must be non-zero
//! - Coordinate transformations must be bijective within grid bounds
//! - Wavenumber computations must respect Nyquist limits

use kwavers_core::error::{ConfigError, KwaversError, KwaversResult};
use leto::Array3;

mod cartesian;
mod cylindrical;
#[cfg(test)]
mod tests;

pub use cartesian::CartesianTopology;
pub use cylindrical::CylindricalTopology;

/// Dimensionality of the simulation domain
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyDimension {
    /// One-dimensional (e.g., axial only)
    One,
    /// Two-dimensional (e.g., axisymmetric r-z, or x-y plane)
    Two,
    /// Three-dimensional (full 3D Cartesian)
    Three,
}

/// Grid topology trait for coordinate system abstraction
///
/// Provides a unified interface for different grid topologies,
/// enabling solvers to be written generically over coordinate systems.
///
/// # Mathematical Foundation
///
/// Each topology must:
/// - Define a mapping from integer indices to physical coordinates
/// - Provide wavenumber grids for spectral methods
/// - Compute metric coefficients (e.g., volume elements)
/// - Support coordinate bounds checking
pub trait GridTopology: Send + Sync {
    /// Get the dimensionality of this topology
    fn dimensionality(&self) -> TopologyDimension;

    /// Get total number of grid points
    fn size(&self) -> usize;

    /// Get grid dimensions as a slice — returns [n0, n1, n2] where unused dimensions should be 1
    fn dimensions(&self) -> [usize; 3];

    /// Get grid spacing as a slice — returns [d0, d1, d2] in meters
    fn spacing(&self) -> [f64; 3];

    /// Get maximum extents in each direction (meters)
    fn extents(&self) -> [f64; 3];

    /// Convert multi-dimensional indices to physical coordinates
    fn indices_to_coordinates(&self, indices: [usize; 3]) -> [f64; 3];

    /// Convert physical coordinates to grid indices — returns `None` if out of bounds
    fn coordinates_to_indices(&self, coords: [f64; 3]) -> Option<[usize; 3]>;

    /// Get the metric coefficient (volume/area element) at given indices
    ///
    /// - Cartesian: dx * dy * dz
    /// - Cylindrical: r * dr * dz (area)
    /// - Spherical: r² * sin(θ) * dr * dθ * dφ
    fn metric_coefficient(&self, indices: [usize; 3]) -> f64;

    /// Check if grid has uniform spacing in all active dimensions
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn is_uniform(&self) -> bool;

    /// Maximum wavenumber supported (Nyquist limit)
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn k_max(&self) -> f64;

    /// Create a zero-initialized field matching this grid topology
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn create_field(&self) -> Array3<f64> {
        let [n0, n1, n2] = self.dimensions();
        Array3::zeros([n0, n1, n2])
    }

    /// Validate that indices are within grid bounds
    /// # Errors
    /// - Returns `KwaversError::Config` if the precondition for a Config-class constraint is violated.
    ///
    fn validate_indices(&self, indices: [usize; 3]) -> KwaversResult<()> {
        let [n0, n1, n2] = self.dimensions();
        let [i, j, k] = indices;

        if i >= n0 || j >= n1 || k >= n2 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "grid indices".to_owned(),
                value: format!("[{}, {}, {}]", i, j, k),
                constraint: format!("Must be within [0..{}, 0..{}, 0..{}]", n0, n1, n2),
            }));
        }

        Ok(())
    }
}
