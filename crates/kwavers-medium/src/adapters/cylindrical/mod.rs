//! Cylindrical medium projection adapter for axisymmetric solvers
//!
//! This module provides an adapter that projects a 3D `Medium` onto a 2D
//! cylindrical coordinate system for use with axisymmetric solvers. The
//! projection maintains mathematical correctness by sampling the medium
//! along the axis of symmetry (θ = 0 plane in cylindrical coordinates).
//!
//! # Mathematical Foundation
//!
//! For axisymmetric problems, the medium properties are independent of the
//! azimuthal angle θ. The projection samples the 3D medium at:
//!
//! ```text
//! (x, y, z) = (r, 0, z)  in Cartesian coordinates
//! (r, θ, z) = (r, 0, z)  in cylindrical coordinates
//! ```
//!
//! # Physical Invariants
//!
//! The projection preserves:
//! - Sound speed bounds: `min(c_3D) ≤ min(c_2D) ≤ max(c_2D) ≤ max(c_3D)`
//! - Homogeneity: Uniform 3D medium → Uniform 2D field
//! - Physical constraints: Positive density, sound speed, non-negative absorption
//!
//! # Module layout
//!
//! - [`construction`]: `new` constructor — samples the 3D medium at the
//!   `θ = 0` plane and caches the resulting 2D property arrays.
//! - [`accessors`]: read-only field views, point-wise samplers, dimension
//!   and spacing queries.
//! - [`validation`]: post-construction physical-bound invariant check
//!   (`validate_projection`).
//!
//! # Example
//!
//! ```rust
//! use kwavers_medium::{HomogeneousMedium, adapters::CylindricalMediumProjection};
//! use kwavers_grid::{Grid, CylindricalTopology};
//!
//! # fn example() -> kwavers_core::error::KwaversResult<()> {
//! let grid = Grid::new(128, 128, 128, 0.0001, 0.0001, 0.0001)?;
//! let medium = HomogeneousMedium::water(&grid);
//!
//! let topology = CylindricalTopology::new(128, 64, 0.0001, 0.0001)?;
//!
//! let projection = CylindricalMediumProjection::new(&medium, &grid, &topology)?;
//!
//! let c_2d = projection.sound_speed_field();  // Shape: (nz, nr)
//! let rho_2d = projection.density_field();
//! # Ok(())
//! # }
//! ```

mod accessors;
mod construction;
mod validation;

#[cfg(test)]
mod tests;

use ndarray::Array2;
use std::fmt;

use crate::Medium;
use kwavers_grid::{CylindricalTopology, Grid};

/// Cylindrical projection of a 3D medium for axisymmetric solvers
///
/// This adapter samples a 3D `Medium` along the θ = 0 plane in cylindrical
/// coordinates, producing 2D property arrays indexed by (z, r).
///
/// # Lifetime
///
/// The projection borrows the medium and grid with lifetime `'a`, ensuring
/// they remain valid for the duration of the projection's use.
///
/// # Caching Strategy
///
/// Property arrays are computed once during construction and cached for
/// efficient repeated access. This is acceptable because:
/// - 2D arrays are much smaller than 3D (typically <1 MB)
/// - Axisymmetric solvers access properties frequently
/// - Construction cost is amortized over many solver iterations
pub struct CylindricalMediumProjection<'a, M: Medium> {
    /// Reference to the 3D medium
    pub(super) medium: &'a M,
    /// Reference to the 3D grid
    pub(super) grid: &'a Grid,
    /// Reference to the cylindrical topology
    pub(super) topology: &'a CylindricalTopology,

    // Cached 2D projections (nz × nr)
    /// Sound speed field (m/s)
    pub(super) sound_speed_2d: Array2<f64>,
    /// Density field (kg/m³)
    pub(super) density_2d: Array2<f64>,
    /// Absorption coefficient field (Np/m)
    pub(super) absorption_2d: Array2<f64>,
    /// Nonlinearity parameter B/A (optional)
    pub(super) nonlinearity_2d: Option<Array2<f64>>,

    // Cached scalar properties
    /// Maximum sound speed in the projected medium
    pub(super) max_sound_speed: f64,
    /// Minimum sound speed in the projected medium
    pub(super) min_sound_speed: f64,
    /// Whether the medium is homogeneous
    pub(super) is_homogeneous: bool,
}

impl<'a, M: Medium> fmt::Debug for CylindricalMediumProjection<'a, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CylindricalMediumProjection")
            .field("dimensions", &(self.topology.nz, self.topology.nr))
            .field("spacing", &(self.topology.dz, self.topology.dr))
            .field("max_sound_speed", &self.max_sound_speed)
            .field("min_sound_speed", &self.min_sound_speed)
            .field("is_homogeneous", &self.is_homogeneous)
            .field("has_nonlinearity", &self.nonlinearity_2d.is_some())
            .finish()
    }
}
