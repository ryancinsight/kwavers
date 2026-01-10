//! Medium adapters for coordinate system projections
//!
//! This module provides adapters that project 3D media onto lower-dimensional
//! coordinate systems for specialized solvers. The adapters maintain mathematical
//! correctness while enabling efficient computation in reduced-dimension spaces.
//!
//! # Available Adapters
//!
//! - [`CylindricalMediumProjection`] - Projects 3D medium onto 2D cylindrical grid
//!   for axisymmetric solvers
//!
//! # Design Philosophy
//!
//! Adapters follow the Adapter pattern from SOLID principles:
//! - **Single Responsibility**: Each adapter handles one projection type
//! - **Open/Closed**: New projections can be added without modifying existing code
//! - **Liskov Substitution**: Projections preserve physical correctness
//! - **Interface Segregation**: Minimal, focused APIs
//! - **Dependency Inversion**: Depend on `Medium` trait, not concrete types
//!
//! # Example
//!
//! ```rust
//! use kwavers::domain::medium::{HomogeneousMedium, adapters::CylindricalMediumProjection};
//! use kwavers::domain::grid::{Grid, CylindricalTopology};
//!
//! # fn example() -> kwavers::domain::core::error::KwaversResult<()> {
//! // Create 3D medium and grid
//! let grid = Grid::new(128, 128, 128, 0.0001, 0.0001, 0.0001)?;
//! let medium = HomogeneousMedium::water(&grid);
//!
//! // Define 2D cylindrical topology
//! let topology = CylindricalTopology::new(128, 64, 0.0001, 0.0001)?;
//!
//! // Project to 2D for axisymmetric solver
//! let projection = CylindricalMediumProjection::new(&medium, &grid, &topology)?;
//!
//! // Use projected fields
//! let sound_speed_2d = projection.sound_speed_field();  // Shape: (nz, nr)
//! # Ok(())
//! # }
//! ```

mod cylindrical;

pub use cylindrical::CylindricalMediumProjection;
