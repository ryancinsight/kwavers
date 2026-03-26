//! Physics state container - Single Source of Truth for all field data
//!
//! This struct owns the field data directly, avoiding unnecessary `Arc<RwLock>`
//! indirection. For concurrent access, wrap the entire PhysicsState in `Arc<RwLock>`
//! at the application level if needed.

use crate::core::error::{KwaversResult, PhysicsError};
use crate::domain::grid::Grid;
use ndarray::{Array3, Array4, ArrayView3, ArrayViewMut3, Axis};

// Use the global field_indices module for consistency
pub use crate::domain::field::indices as field_indices;

/// Physics state container - Single Source of Truth for all field data
#[derive(Debug)]
pub struct PhysicsState {
    /// Main 4D field array containing all physics quantities
    fields: Array4<f64>,

    /// Grid dimensions
    grid: Grid,
}

impl PhysicsState {
    /// Create a new physics state with the given grid
    pub fn new(grid: Grid) -> Self {
        let (nx, ny, nz) = grid.dimensions();
        let fields = Array4::<f64>::zeros((field_indices::TOTAL_FIELDS, nx, ny, nz));

        Self {
            fields,
            grid,
        }
    }

    /// Get a read-only view of a specific field (zero-copy)
    pub fn get_field(&self, field_index: usize) -> KwaversResult<ArrayView3<'_, f64>> {
        if field_index >= field_indices::TOTAL_FIELDS {
            return Err(PhysicsError::InvalidFieldIndex(field_index).into());
        }

        Ok(self.fields.index_axis(Axis(0), field_index))
    }

    /// Get a mutable view of a specific field (zero-copy)
    pub fn get_field_mut(&mut self, field_index: usize) -> KwaversResult<ArrayViewMut3<'_, f64>> {
        if field_index >= field_indices::TOTAL_FIELDS {
            return Err(PhysicsError::InvalidFieldIndex(field_index).into());
        }

        Ok(self.fields.index_axis_mut(Axis(0), field_index))
    }

    /// Apply a closure to a field for reading (zero-copy)
    pub fn with_field<F, R>(&self, field_index: usize, f: F) -> KwaversResult<R>
    where
        F: FnOnce(ArrayView3<f64>) -> R,
    {
        if field_index >= field_indices::TOTAL_FIELDS {
            return Err(PhysicsError::InvalidFieldIndex(field_index).into());
        }

        Ok(f(self.fields.index_axis(Axis(0), field_index)))
    }

    /// Apply a closure to a field for mutation (zero-copy)
    pub fn with_field_mut<F, R>(&mut self, field_index: usize, f: F) -> KwaversResult<R>
    where
        F: FnOnce(ArrayViewMut3<f64>) -> R,
    {
        if field_index >= field_indices::TOTAL_FIELDS {
            return Err(PhysicsError::InvalidFieldIndex(field_index).into());
        }

        Ok(f(self.fields.index_axis_mut(Axis(0), field_index)))
    }

    /// Get a cloned copy of a field (allocates memory)
    /// Prefer `get_field()` for zero-copy access when possible
    pub fn clone_field(&self, field_index: usize) -> KwaversResult<Array3<f64>> {
        self.with_field(field_index, |field| field.to_owned())
    }

    /// Update a specific field with new data
    pub fn update_field(&mut self, field_index: usize, data: &Array3<f64>) -> KwaversResult<()> {
        if field_index >= field_indices::TOTAL_FIELDS {
            return Err(PhysicsError::InvalidFieldIndex(field_index).into());
        }

        let mut field = self.fields.index_axis_mut(Axis(0), field_index);
        if data.shape() != field.shape() {
            return Err(PhysicsError::DimensionMismatch.into());
        }
        field.assign(data);
        Ok(())
    }

    /// Get direct access to all fields (for plugin system)
    pub fn with_all_fields_mut<F, R>(&mut self, f: F) -> KwaversResult<R>
    where
        F: FnOnce(&mut Array4<f64>) -> R,
    {
        Ok(f(&mut self.fields))
    }

    /// Get all fields - returns a view to avoid cloning
    pub fn get_all_fields(&self) -> &Array4<f64> {
        &self.fields
    }

    /// Get field metadata name
    pub fn get_field_name(&self, field_index: usize) -> &'static str {
        field_indices::field_name(field_index)
    }

    /// Get field metadata unit
    pub fn get_field_unit(&self, field_index: usize) -> &'static str {
        field_indices::field_unit(field_index)
    }

    /// Initialize field with a constant value
    pub fn initialize_field(&mut self, field_index: usize, value: f64) -> KwaversResult<()> {
        self.with_field_mut(field_index, |mut field| {
            field.fill(value);
        })
    }

    /// Initialize field with a function
    pub fn initialize_field_with<F>(&mut self, field_index: usize, init_fn: F) -> KwaversResult<()>
    where
        F: Fn(usize, usize, usize) -> f64,
    {
        self.with_field_mut(field_index, |mut field| {
            let (nx, ny, nz) = (field.shape()[0], field.shape()[1], field.shape()[2]);
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        field[[i, j, k]] = init_fn(i, j, k);
                    }
                }
            }
        })
    }

    /// Get grid reference
    pub fn grid(&self) -> &Grid {
        &self.grid
    }
}
