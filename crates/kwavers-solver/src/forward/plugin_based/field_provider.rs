//! Field provider for controlled field access
//!
//! Implements the Principle of Least Privilege by restricting plugin access
//! to only the fields they need.

use super::field_registry::FieldRegistry;
use kwavers_core::error::FieldError;
use kwavers_domain::field::mapping::UnifiedFieldType;
use ndarray::{ArrayView3, ArrayViewMut3};

/// Field provider for plugins with restricted access
#[derive(Debug)]
pub struct FieldProvider<'a> {
    registry: &'a mut FieldRegistry,
    allowed_fields: Vec<UnifiedFieldType>,
}

impl<'a> FieldProvider<'a> {
    /// Create a new field provider with restricted access
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(registry: &'a mut FieldRegistry, allowed_fields: Vec<UnifiedFieldType>) -> Self {
        Self {
            registry,
            allowed_fields,
        }
    }

    /// Get a field view (zero-copy, read-only)
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn get_field(
        &self,
        field_type: UnifiedFieldType,
    ) -> Result<ArrayView3<'_, f64>, FieldError> {
        self.check_permission(field_type)?;
        self.registry.get_field(field_type)
    }

    /// Get a mutable field view (zero-copy)
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn get_field_mut(
        &mut self,
        field_type: UnifiedFieldType,
    ) -> Result<ArrayViewMut3<'_, f64>, FieldError> {
        self.check_permission(field_type)?;
        self.registry.get_field_mut(field_type)
    }

    /// Check if a field is available to this provider
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn has_field(&self, field_type: UnifiedFieldType) -> bool {
        self.allowed_fields.contains(&field_type) && self.registry.has_field(field_type)
    }

    /// Get list of fields available to this provider
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn available_fields(&self) -> Vec<UnifiedFieldType> {
        self.allowed_fields
            .iter()
            .filter(|&&ft| self.registry.has_field(ft))
            .copied()
            .collect()
    }

    /// Check if provider has permission to access a field
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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
    use kwavers_grid::Grid;

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
