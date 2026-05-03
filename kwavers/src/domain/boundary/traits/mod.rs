//! Boundary condition trait system for unified field handling.

pub mod core;
pub mod layer;
#[cfg(test)]
mod tests;
pub mod types;

pub use core::{AbsorbingBoundary, BoundaryCondition, PeriodicBoundary, ReflectiveBoundary};
pub use layer::{BoundaryLayer, BoundaryLayerManager};
pub use types::{BoundaryDirections, BoundaryDomain, FieldType};
