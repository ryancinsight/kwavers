//! Field registry for dynamic field management
//!
//! Provides type-safe field management with O(1) access using direct indexing.
//! Follows SOLID principles with single responsibility for field storage.

use crate::error::{FieldError, KwaversResult};
use crate::grid::Grid;
use crate::physics::field_mapping::UnifiedFieldType;
use log::{debug, info};
use ndarray::{Array3, Array4, ArrayView3, ArrayViewMut3, Axis};

#[derive(Clone, Debug)]
struct FieldMetadata {
    /// Index in the Array4
    index: usize,
    /// Field description
    description: String,
    /// Whether field is currently active
    active: bool,
}

/// Dynamic field registry for type-safe field management
#[derive(Debug)]
pub struct FieldRegistry {
    /// Registered fields indexed by `UnifiedFieldType` numeric value
    fields: Vec<Option<FieldMetadata>>,
    /// Field data storage - dynamically sized
    data: Option<Array4<f64>>,
    /// Grid dimensions for validation
    grid_dims: (usize, usize, usize),
    /// Flag to defer allocation until `build()` is called
    deferred_allocation: bool,
    /// Counter for assigned indices in data array
    next_data_index: usize,
}

impl FieldRegistry {
    /// Create a new field registry with deferred allocation
    pub fn new(grid: &Grid) -> Self {
        Self {
            fields: vec![None; UnifiedFieldType::COUNT],
            data: None,
            grid_dims: (grid.nx, grid.ny, grid.nz),
            deferred_allocation: true,
            next_data_index: 0,
        }
    }

    /// Build the field registry by allocating data array
    pub fn build(&mut self) -> KwaversResult<()> {
        let max_field_index = self.fields.len();
        if max_field_index == 0 {
            self.data = None;
            self.deferred_allocation = false;
            return Ok(());
        }

        let (nx, ny, nz) = self.grid_dims;
        self.data = Some(Array4::zeros((max_field_index, nx, ny, nz)));
        self.deferred_allocation = false;

        debug!(
            "Built FieldRegistry: {} fields, dimensions ({}, {}, {})",
            self.next_data_index, nx, ny, nz
        );
        Ok(())
    }

    /// Register a new field dynamically
    pub fn register_field(
        &mut self,
        field_type: UnifiedFieldType,
        description: String,
    ) -> KwaversResult<()> {
        let idx = field_type as usize;

        // Check if already registered
        if idx < self.fields.len() && self.fields[idx].is_some() {
            return Ok(());
        }

        // Ensure Vec is large enough
        while self.fields.len() <= idx {
            self.fields.push(None);
        }

        self.fields[idx] = Some(FieldMetadata {
            index: idx,
            description,
            active: true,
        });
        self.next_data_index += 1;

        // Reallocate if not using deferred allocation
        if !self.deferred_allocation {
            self.reallocate_data()?;
        }

        info!("Registered field: {} at index {}", field_type, idx);
        Ok(())
    }

    /// Register multiple fields at once
    pub fn register_fields(&mut self, fields: &[(UnifiedFieldType, String)]) -> KwaversResult<()> {
        for (field_type, description) in fields {
            self.register_field(*field_type, description.clone())?;
        }
        Ok(())
    }

    /// Get a field view (zero-copy, read-only)
    pub fn get_field(
        &self,
        field_type: UnifiedFieldType,
    ) -> Result<ArrayView3<'_, f64>, FieldError> {
        let metadata = self.get_metadata(field_type)?;

        if !metadata.active {
            return Err(FieldError::Inactive(field_type.name().to_string()));
        }

        let data = self.data.as_ref().ok_or(FieldError::DataNotInitialized)?;

        Ok(data.index_axis(Axis(0), metadata.index))
    }

    /// Get a mutable field view (zero-copy)
    pub fn get_field_mut(
        &mut self,
        field_type: UnifiedFieldType,
    ) -> Result<ArrayViewMut3<'_, f64>, FieldError> {
        // Get metadata info without borrowing self
        let idx = field_type as usize;
        let (metadata_index, is_active) = self
            .fields
            .get(idx)
            .and_then(|opt| opt.as_ref())
            .map(|m| (m.index, m.active))
            .ok_or_else(|| FieldError::NotRegistered(field_type.name().to_string()))?;

        if !is_active {
            return Err(FieldError::Inactive(field_type.name().to_string()));
        }

        let data = self.data.as_mut().ok_or(FieldError::DataNotInitialized)?;

        Ok(data.index_axis_mut(Axis(0), metadata_index))
    }

    /// Set a specific field with dimension validation
    pub fn set_field(
        &mut self,
        field_type: UnifiedFieldType,
        values: &Array3<f64>,
    ) -> KwaversResult<()> {
        // Validate dimensions
        let actual_dims = values.dim();
        if actual_dims != self.grid_dims {
            return Err(FieldError::DimensionMismatch {
                field: field_type.name().to_string(),
                expected: self.grid_dims,
                actual: actual_dims,
            }
            .into());
        }

        let mut field_view = self.get_field_mut(field_type)?;
        field_view.assign(values);
        Ok(())
    }

    /// Check if a field is registered
    #[must_use]
    pub fn has_field(&self, field_type: UnifiedFieldType) -> bool {
        let idx = field_type as usize;
        idx < self.fields.len() && self.fields[idx].is_some()
    }

    /// Get list of registered fields
    #[must_use]
    pub fn registered_fields(&self) -> Vec<UnifiedFieldType> {
        (0..UnifiedFieldType::COUNT)
            .filter_map(|i| {
                if i < self.fields.len() && self.fields[i].is_some() {
                    UnifiedFieldType::from_index(i)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Clear all field data (set to zero)
    pub fn clear_fields(&mut self) {
        if let Some(ref mut data) = self.data {
            data.fill(0.0);
        }
    }

    /// Get the number of registered fields
    #[must_use]
    pub fn num_fields(&self) -> usize {
        self.next_data_index
    }

    /// Get grid dimensions
    #[must_use]
    pub fn grid_dims(&self) -> (usize, usize, usize) {
        self.grid_dims
    }

    /// Get direct access to data array (for advanced operations)
    #[must_use]
    pub fn data(&self) -> Option<&Array4<f64>> {
        self.data.as_ref()
    }

    /// Get mutable direct access to data array
    pub fn data_mut(&mut self) -> Option<&mut Array4<f64>> {
        self.data.as_mut()
    }

    // Private helper methods

    fn get_metadata(&self, field_type: UnifiedFieldType) -> Result<&FieldMetadata, FieldError> {
        self.fields
            .get(field_type as usize)
            .and_then(|opt| opt.as_ref())
            .ok_or_else(|| FieldError::NotRegistered(field_type.name().to_string()))
    }

    fn reallocate_data(&mut self) -> KwaversResult<()> {
        let num_fields = self.fields.iter().filter(|f| f.is_some()).count();
        if num_fields == 0 {
            self.data = None;
            return Ok(());
        }

        let (nx, ny, nz) = self.grid_dims;
        let mut resized_data = Array4::zeros((self.fields.len(), nx, ny, nz));

        // Copy existing data if present
        if let Some(existing_data) = &self.data {
            let min_fields = existing_data.shape()[0].min(resized_data.shape()[0]);
            for i in 0..min_fields {
                resized_data
                    .index_axis_mut(Axis(0), i)
                    .assign(&existing_data.index_axis(Axis(0), i));
            }
        }

        self.data = Some(resized_data);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_registry() {
        let grid = Grid::new(10, 10, 10, 1.0, 1.0, 1.0).unwrap();
        let mut registry = FieldRegistry::new(&grid);

        // Register fields
        registry
            .register_field(UnifiedFieldType::Pressure, "Pressure field".to_string())
            .unwrap();

        registry
            .register_field(
                UnifiedFieldType::Temperature,
                "Temperature field".to_string(),
            )
            .unwrap();

        // Build the registry
        registry.build().unwrap();

        // Check fields are registered
        assert!(registry.has_field(UnifiedFieldType::Pressure));
        assert!(registry.has_field(UnifiedFieldType::Temperature));
        assert_eq!(registry.num_fields(), 2);

        // Test field access
        let pressure = registry.get_field(UnifiedFieldType::Pressure).unwrap();
        assert_eq!(pressure.shape(), &[10, 10, 10]);
    }
}
