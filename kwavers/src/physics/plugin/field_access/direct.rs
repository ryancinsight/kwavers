use crate::core::error::{KwaversResult, PhysicsError};
use crate::domain::field::mapping::UnifiedFieldType;
use ndarray::{Array4, ArrayView3, ArrayViewMut3};
use std::collections::HashSet;

/// This provides direct access to the fields array but with compile-time
/// guarantees about which fields can be accessed
#[derive(Debug)]
pub struct DirectPluginFieldAccess<'a> {
    /// Direct reference to fields array
    fields: &'a mut Array4<f64>,
    /// Fields this plugin is allowed to read
    readable_indices: HashSet<usize>,
    /// Fields this plugin is allowed to write
    writable_indices: HashSet<usize>,
}

impl<'a> DirectPluginFieldAccess<'a> {
    /// Create a new direct field accessor
    pub fn new(
        fields: &'a mut Array4<f64>,
        required_fields: &[UnifiedFieldType],
        provided_fields: &[UnifiedFieldType],
    ) -> Self {
        let mut readable_indices = HashSet::new();
        let mut writable_indices = HashSet::new();

        // Plugin can read its required fields
        for field in required_fields {
            readable_indices.insert(field.index());
        }

        // Plugin can read and write its provided fields
        for field in provided_fields {
            readable_indices.insert(field.index());
            writable_indices.insert(field.index());
        }

        Self {
            fields,
            readable_indices,
            writable_indices,
        }
    }

    /// Get a read-only view of a field
    pub fn get_field(&self, field: UnifiedFieldType) -> KwaversResult<ArrayView3<'_, f64>> {
        let index = field.index();
        if !self.readable_indices.contains(&index) {
            return Err(PhysicsError::UnauthorizedFieldAccess {
                field: field.name().to_string(),
                operation: "read".to_string(),
            }
            .into());
        }

        Ok(self.fields.index_axis(ndarray::Axis(0), index))
    }

    /// Get a mutable view of a field
    pub fn get_field_mut(
        &mut self,
        field: UnifiedFieldType,
    ) -> KwaversResult<ArrayViewMut3<'_, f64>> {
        let index = field.index();
        if !self.writable_indices.contains(&index) {
            return Err(PhysicsError::UnauthorizedFieldAccess {
                field: field.name().to_string(),
                operation: "write".to_string(),
            }
            .into());
        }

        Ok(self.fields.index_axis_mut(ndarray::Axis(0), index))
    }
}
