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

mod projection;

pub use projection::CylindricalMediumProjection;
