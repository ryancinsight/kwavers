use crate::core::error::{KwaversResult, PhysicsError};
use crate::domain::field::mapping::UnifiedFieldType;
use crate::physics::acoustics::state::{FieldView, PhysicsState};
use ndarray::{ArrayView3, ArrayViewMut3};
use std::collections::HashSet;

/// Mutable field accessor for plugins
#[derive(Debug)]
pub struct PluginFieldAccessMut<'a> {
    /// Mutable reference to the physics state
    state: &'a mut PhysicsState,
    /// Fields this plugin is allowed to read
    readable_fields: HashSet<UnifiedFieldType>,
    /// Fields this plugin is allowed to write
    writable_fields: HashSet<UnifiedFieldType>,
}

impl<'a> PluginFieldAccessMut<'a> {
    /// Create a new mutable field accessor for a plugin
    pub fn new(
        state: &'a mut PhysicsState,
        required_fields: &[UnifiedFieldType],
        provided_fields: &[UnifiedFieldType],
    ) -> Self {
        let mut readable_fields = HashSet::new();
        let mut writable_fields = HashSet::new();

        // Plugin can read its required fields
        for field in required_fields {
            readable_fields.insert(*field);
        }

        // Plugin can read and write its provided fields
        for field in provided_fields {
            readable_fields.insert(*field);
            writable_fields.insert(*field);
        }

        Self {
            state,
            readable_fields,
            writable_fields,
        }
    }

    /// Check if plugin can read a field
    #[must_use]
    pub fn can_read(&self, field: UnifiedFieldType) -> bool {
        self.readable_fields.contains(&field)
    }

    /// Check if plugin can write a field
    #[must_use]
    pub fn can_write(&self, field: UnifiedFieldType) -> bool {
        self.writable_fields.contains(&field)
    }

    /// Get read access to a field
    pub fn get_field(&self, field: UnifiedFieldType) -> KwaversResult<FieldView<'_>> {
        if !self.can_read(field) {
            return Err(PhysicsError::UnauthorizedFieldAccess {
                field: field.name().to_string(),
                operation: "read".to_string(),
            }
            .into());
        }

        self.state.get_field(field.index())
    }

    /// Get mutable access to a field
    pub fn get_field_mut(
        &mut self,
        field: UnifiedFieldType,
    ) -> KwaversResult<ArrayViewMut3<'_, f64>> {
        if !self.can_write(field) {
            return Err(PhysicsError::UnauthorizedFieldAccess {
                field: field.name().to_string(),
                operation: "write".to_string(),
            }
            .into());
        }

        self.state.get_field_mut(field.index())
    }

    /// Apply a closure to a mutable field
    pub fn with_field_mut<F, R>(&mut self, field: UnifiedFieldType, f: F) -> KwaversResult<R>
    where
        F: FnOnce(ArrayViewMut3<f64>) -> R,
    {
        if !self.can_write(field) {
            return Err(PhysicsError::UnauthorizedFieldAccess {
                field: field.name().to_string(),
                operation: "write".to_string(),
            }
            .into());
        }

        self.state.with_field_mut(field.index(), f)
    }

    /// Apply a closure to a readable field
    pub fn with_field<F, R>(&self, field: UnifiedFieldType, f: F) -> KwaversResult<R>
    where
        F: FnOnce(ArrayView3<f64>) -> R,
    {
        if !self.can_read(field) {
            return Err(PhysicsError::UnauthorizedFieldAccess {
                field: field.name().to_string(),
                operation: "read".to_string(),
            }
            .into());
        }

        self.state.with_field(field.index(), f)
    }
}
