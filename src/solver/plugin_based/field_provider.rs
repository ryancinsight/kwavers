//! Field provider for controlled field access
//!
//! Implements the Principle of Least Privilege by restricting plugin access
//! to only the fields they need.

use super::field_registry::FieldRegistry;
use crate::error::FieldError;
use crate::physics::field_mapping::UnifiedFieldType;
use ndarray::{ArrayView3, ArrayViewMut3};

/// Field provider for plugins with restricted access
#[derive(Debug)]
pub struct FieldProvider<'a> {
    registry: &'a mut FieldRegistry,
    allowed_fields: Vec<UnifiedFieldType>,
}

impl<'a> FieldProvider<'a> {
    /// Create a new field provider with restricted access
    pub fn new(registry: &'a mut FieldRegistry, allowed_fields: Vec<UnifiedFieldType>) -> Self {
        Self {
            registry,
            allowed_fields,
        }
    }

    /// Get a field view (zero-copy, read-only)
    pub fn get_field(
        &self,
        field_type: UnifiedFieldType,
    ) -> Result<ArrayView3<'_, f64>, FieldError> {
        self.check_permission(field_type)?;
        self.registry.get_field(field_type)
    }

    /// Get a mutable field view (zero-copy)
    pub fn get_field_mut(
        &mut self,
        field_type: UnifiedFieldType,
    ) -> Result<ArrayViewMut3<'_, f64>, FieldError> {
        self.check_permission(field_type)?;
        self.registry.get_field_mut(field_type)
    }

    /// Check if a field is available to this provider
    #[must_use]
    pub fn has_field(&self, field_type: UnifiedFieldType) -> bool {
        self.allowed_fields.contains(&field_type) && self.registry.has_field(field_type)
    }

    /// Get list of fields available to this provider
    #[must_use]
    pub fn available_fields(&self) -> Vec<UnifiedFieldType> {
        self.allowed_fields
            .iter()
            .filter(|&&ft| self.registry.has_field(ft))
            .copied()
            .collect()
    }

    /// Check if provider has permission to access a field
    fn check_permission(&self, field_type: UnifiedFieldType) -> Result<(), FieldError> {
        if !self.allowed_fields.contains(&field_type) {
            return Err(FieldError::NotRegistered(format!(
                "Field {} not allowed for this plugin",
                field_type.name()
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;

    #[test]
    fn test_field_provider_access_control() {
        let grid = Grid::new(10, 10, 10, 1.0, 1.0, 1.0).unwrap();
        let mut registry = FieldRegistry::new(&grid);

        // Register multiple fields
        registry.register_field(UnifiedFieldType::Pressure).unwrap();
        registry
            .register_field(UnifiedFieldType::Temperature)
            .unwrap();
        registry.build().unwrap();

        // Create provider with limited access
        let allowed = vec![UnifiedFieldType::Pressure];
        let provider = FieldProvider::new(&mut registry, allowed);

        // Should access allowed field
        assert!(provider.has_field(UnifiedFieldType::Pressure));

        // Should not access disallowed field
        assert!(!provider.has_field(UnifiedFieldType::Temperature));

        // Should fail to get disallowed field
        assert!(provider.get_field(UnifiedFieldType::Temperature).is_err());
    }
}
