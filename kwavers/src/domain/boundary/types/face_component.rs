//! `BoundaryFace` and `BoundaryComponent` enums.

use serde::{Deserialize, Serialize};

/// Boundary face specification for rectangular domains.
///
/// Identifies which face of a 3D rectangular domain the boundary
/// condition applies to. For 2D problems, ZMin/ZMax are unused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BoundaryFace {
    /// Minimum x-face (x = x_min)
    XMin,
    /// Maximum x-face (x = x_max)
    XMax,
    /// Minimum y-face (y = y_min)
    YMin,
    /// Maximum y-face (y = y_max)
    YMax,
    /// Minimum z-face (z = z_min)
    ZMin,
    /// Maximum z-face (z = z_max)
    ZMax,
}

/// Boundary component specification for vector fields.
///
/// Specifies which components of a vector field the boundary condition
/// applies to. Essential for elastic waves (displacement vector) and
/// electromagnetics (E and H field vectors).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BoundaryComponent {
    /// All vector components
    All,
    /// X-component only
    X,
    /// Y-component only
    Y,
    /// Z-component only
    Z,
    /// Normal component (n·u)
    Normal,
    /// Tangential components (u - (n·u)n)
    Tangential,
}
