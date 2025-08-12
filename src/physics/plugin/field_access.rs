//! Safe field access for plugins
//!
//! This module provides a safe API for plugins to access only the fields
//! they have declared as required or provided, preventing accidental access
//! to unrelated fields and improving encapsulation.

use crate::physics::field_mapping::UnifiedFieldType;
use crate::physics::state::{PhysicsState, FieldReadGuard, FieldWriteGuard};
use crate::error::{KwaversResult, PhysicsError};
use ndarray::{Array4, ArrayView3, ArrayViewMut3};
use std::collections::HashSet;

/// Safe field accessor for plugins
/// 
/// This struct ensures plugins can only access fields they've declared
pub struct PluginFieldAccess<'a> {
    /// Reference to the physics state
    state: &'a PhysicsState,
    /// Fields this plugin is allowed to read
    readable_fields: HashSet<UnifiedFieldType>,
    /// Fields this plugin is allowed to write
    writable_fields: HashSet<UnifiedFieldType>,
}

impl<'a> PluginFieldAccess<'a> {
    /// Create a new field accessor for a plugin
    pub fn new(
        state: &'a PhysicsState,
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
    pub fn can_read(&self, field: UnifiedFieldType) -> bool {
        self.readable_fields.contains(&field)
    }
    
    /// Check if plugin can write a field
    pub fn can_write(&self, field: UnifiedFieldType) -> bool {
        self.writable_fields.contains(&field)
    }
    
    /// Get read access to a field
    pub fn get_field(&self, field: UnifiedFieldType) -> KwaversResult<FieldReadGuard> {
        if !self.can_read(field) {
            return Err(PhysicsError::UnauthorizedFieldAccess {
                field: field.name().to_string(),
                operation: "read".to_string(),
            }.into());
        }
        
        self.state.get_field(field.index())
    }
    
    /// Get write access to a field
    pub fn get_field_mut(&self, field: UnifiedFieldType) -> KwaversResult<FieldWriteGuard> {
        if !self.can_write(field) {
            return Err(PhysicsError::UnauthorizedFieldAccess {
                field: field.name().to_string(),
                operation: "write".to_string(),
            }.into());
        }
        
        self.state.get_field_mut(field.index())
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
            }.into());
        }
        
        self.state.with_field(field.index(), f)
    }
    
    /// Apply a closure to a writable field
    pub fn with_field_mut<F, R>(&self, field: UnifiedFieldType, f: F) -> KwaversResult<R>
    where
        F: FnOnce(ArrayViewMut3<f64>) -> R,
    {
        if !self.can_write(field) {
            return Err(PhysicsError::UnauthorizedFieldAccess {
                field: field.name().to_string(),
                operation: "write".to_string(),
            }.into());
        }
        
        self.state.with_field_mut(field.index(), f)
    }
}

/// Direct field access for high-performance plugins
/// 
/// This provides direct access to the fields array but with compile-time
/// guarantees about which fields can be accessed
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
    pub fn get_field(&self, field: UnifiedFieldType) -> KwaversResult<ArrayView3<f64>> {
        let index = field.index();
        if !self.readable_indices.contains(&index) {
            return Err(PhysicsError::UnauthorizedFieldAccess {
                field: field.name().to_string(),
                operation: "read".to_string(),
            }.into());
        }
        
        Ok(self.fields.index_axis(ndarray::Axis(0), index))
    }
    
    /// Get a mutable view of a field
    pub fn get_field_mut(&mut self, field: UnifiedFieldType) -> KwaversResult<ArrayViewMut3<f64>> {
        let index = field.index();
        if !self.writable_indices.contains(&index) {
            return Err(PhysicsError::UnauthorizedFieldAccess {
                field: field.name().to_string(),
                operation: "write".to_string(),
            }.into());
        }
        
        Ok(self.fields.index_axis_mut(ndarray::Axis(0), index))
    }
    
    /// Get the raw fields array (for legacy compatibility)
    /// 
    /// **WARNING**: This bypasses access control. Use only when necessary.
    #[deprecated(since = "1.6.0", note = "Use get_field/get_field_mut for safe access")]
    pub fn raw_fields(&mut self) -> &mut Array4<f64> {
        self.fields
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    
    #[test]
    fn test_plugin_field_access() {
        let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1);
        let state = PhysicsState::new(grid);
        
        // Create accessor with specific permissions
        let required = vec![UnifiedFieldType::Pressure];
        let provided = vec![UnifiedFieldType::Temperature];
        let access = PluginFieldAccess::new(&state, &required, &provided);
        
        // Should be able to read pressure (required)
        assert!(access.can_read(UnifiedFieldType::Pressure));
        assert!(!access.can_write(UnifiedFieldType::Pressure));
        
        // Should be able to read and write temperature (provided)
        assert!(access.can_read(UnifiedFieldType::Temperature));
        assert!(access.can_write(UnifiedFieldType::Temperature));
        
        // Should not be able to access density (not declared)
        assert!(!access.can_read(UnifiedFieldType::Density));
        assert!(!access.can_write(UnifiedFieldType::Density));
    }
    
    #[test]
    fn test_unauthorized_access() {
        let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1);
        let state = PhysicsState::new(grid);
        
        // Create accessor with limited permissions
        let required = vec![UnifiedFieldType::Pressure];
        let provided = vec![];
        let access = PluginFieldAccess::new(&state, &required, &provided);
        
        // Try to write to a read-only field
        let result = access.get_field_mut(UnifiedFieldType::Pressure);
        assert!(result.is_err());
        
        // Try to access an undeclared field
        let result = access.get_field(UnifiedFieldType::Density);
        assert!(result.is_err());
    }
}