//! Physics state traits and type aliases

use super::container::PhysicsState;
use crate::core::error::KwaversResult;
use ndarray::{Array3, ArrayView3, ArrayViewMut3};

/// Direct field view for zero-copy read access
pub type FieldView<'a> = ArrayView3<'a, f64>;

/// Direct mutable field view for zero-copy write access
pub type FieldViewMut<'a> = ArrayViewMut3<'a, f64>;

/// Trait for types that provide access to physics state
pub trait HasPhysicsState {
    /// Get reference to the physics state
    fn physics_state(&self) -> &PhysicsState;

    /// Get mutable reference to the physics state
    fn physics_state_mut(&mut self) -> &mut PhysicsState;

    /// Get a specific field by index
    fn get_field(&self, field_index: usize) -> KwaversResult<FieldView<'_>> {
        self.physics_state().get_field(field_index)
    }

    /// Update a specific field
    fn update_field(&mut self, field_index: usize, data: &Array3<f64>) -> KwaversResult<()> {
        self.physics_state_mut().update_field(field_index, data)
    }
}
