// FDTD Numerics Module
pub mod boundary_stencils;
pub mod finite_difference;
pub mod interpolation;
pub mod staggered_grid;

pub use boundary_stencils::BoundaryStencils;
pub use finite_difference::FiniteDifference;
pub use interpolation::StaggeredInterpolation;
pub use staggered_grid::{FieldComponent, StaggeredGrid};
