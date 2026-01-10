//! Safe field access for domain plugins
//!
//! This module provides a safe API for domain plugins to access field data,
//! ensuring proper encapsulation and preventing accidental access to unrelated fields.

use crate::core::error::KwaversResult;
use crate::domain::field::mapping::UnifiedFieldType;
use ndarray::{Array3, ArrayView3, ArrayViewMut3};
use std::collections::HashSet;

/// Plugin field data container
#[derive(Debug, Clone)]
pub struct PluginFields {
    data: Array3<f64>,
}

impl PluginFields {
    #[must_use]
    pub fn new(data: Array3<f64>) -> Self {
        Self { data }
    }

    #[must_use]
    pub fn view(&self) -> ArrayView3<'_, f64> {
        self.data.view()
    }

    #[must_use]
    pub fn view_mut(&mut self) -> ArrayViewMut3<'_, f64> {
        self.data.view_mut()
    }
}

/// Safe field accessor for plugins
#[derive(Debug)]
pub struct FieldAccessor<'a> {
    fields: &'a PluginFields,
    provided_fields: &'a HashSet<UnifiedFieldType>,
    required_fields: &'a HashSet<UnifiedFieldType>,
}

impl<'a> FieldAccessor<'a> {
    #[must_use]
    pub fn new(
        fields: &'a PluginFields,
        provided_fields: &'a HashSet<UnifiedFieldType>,
        required_fields: &'a HashSet<UnifiedFieldType>,
    ) -> Self {
        Self {
            fields,
            provided_fields,
            required_fields,
        }
    }

    /// Get read-only access to a field
    pub fn get_field(&self, field_type: UnifiedFieldType) -> KwaversResult<ArrayView3<'_, f64>> {
        if !self.can_access_field(field_type) {
            return Err(crate::core::error::FieldError::InvalidFieldAccess {
                field: format!("{:?}", field_type),
                reason: "Field not declared as provided or required".to_string(),
            }.into());
        }

        // For now, return the entire field data
        // In practice, this should return the specific field slice
        Ok(self.fields.view())
    }

    /// Check if a field can be accessed
    #[must_use]
    pub fn can_access_field(&self, field_type: UnifiedFieldType) -> bool {
        self.provided_fields.contains(&field_type) || self.required_fields.contains(&field_type)
    }

    /// Get the set of accessible fields
    #[must_use]
    pub fn accessible_fields(&self) -> HashSet<UnifiedFieldType> {
        self.provided_fields.union(self.required_fields).cloned().collect()
    }
}