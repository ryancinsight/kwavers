//! Canonical Boundary Condition Types — Single Source of Truth.
//!
//! All other modules MUST import from here; no duplicate definitions allowed.
//!
//! ## Mathematical Foundations
//!
//! General boundary condition form: `α·u + β·∂u/∂n = g(x, t)`
//!
//! Special cases:
//! - **Dirichlet** (α=1, β=0): `u = g`
//! - **Neumann** (α=0, β=1): `∂u/∂n = g`
//! - **Robin** (α≠0, β≠0): Mixed condition
//! - **Periodic**: `u(x_min) = u(x_max)` with phase matching
//! - **Absorbing**: Non-reflecting (PML, ABC, Sommerfeld)

pub mod boundary_type;
pub mod conversions;
pub mod domain_specific;
pub mod face_component;
pub mod spec;
#[cfg(test)]
mod tests;

pub use boundary_type::BoundaryType;
pub use domain_specific::{AcousticBoundaryType, ElasticBoundaryType, ElectromagneticBoundaryType};
pub use face_component::{BoundaryFace, FaceBoundaryComponent};
pub use spec::BoundarySpec;
