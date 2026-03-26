//! Centralized physics state management following SOLID principles
//!
//! This module provides a single source of truth for physics field states,
//! eliminating the need for dummy fields scattered across implementations.

pub mod container;
pub mod traits;

pub use container::{field_indices, PhysicsState};
pub use traits::{FieldView, FieldViewMut, HasPhysicsState};

#[cfg(test)]
mod tests;
