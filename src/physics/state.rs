//! Centralized physics state management following SOLID principles
//!
//! This module provides a single source of truth for physics field states,
//! eliminating the need for dummy fields scattered across implementations.

use crate::error::{KwaversResult, PhysicsError};
use crate::grid::Grid;
use ndarray::{Array3, Array4, ArrayView3, ArrayViewMut3, Axis};
use std::collections::HashMap;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

// Use the global field_indices module for consistency
pub use crate::physics::field_indices;

/// Physics state container - Single Source of Truth for all field data
#[derive(Clone, Debug)]
pub struct PhysicsState {
    /// Main 4D field array containing all physics quantities
    fields: Arc<RwLock<Array4<f64>>>,

    /// Grid dimensions
    grid: Grid,

    /// Field metadata
    field_names: HashMap<usize, String>,
    field_units: HashMap<usize, String>,
}

/// RAII guard for read-only field access
/// This struct owns the data it needs to avoid unsafe transmutes
pub struct FieldReadGuard<'a> {
    data: Array3<f64>,
    _guard: RwLockReadGuard<'a, Array4<f64>>,
}

impl<'a> FieldReadGuard<'a> {
    /// Create a new FieldReadGuard
    fn new(guard: RwLockReadGuard<'a, Array4<f64>>, field_index: usize) -> Self {
        let data = guard.index_axis(Axis(0), field_index).to_owned();
        Self {
            data,
            _guard: guard,
        }
    }

    /// Get the field view
    pub fn view(&self) -> ArrayView3<'_, f64> {
        self.data.view()
    }

    /// Convert to owned array
    pub fn to_owned(&self) -> Array3<f64> {
        self.data.clone()
    }
}

impl<'a> std::ops::Deref for FieldReadGuard<'a> {
    type Target = Array3<f64>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/// RAII guard for mutable field access
pub struct FieldWriteGuard<'a> {
    guard: RwLockWriteGuard<'a, Array4<f64>>,
    field_index: usize,
}

impl<'a> FieldWriteGuard<'a> {
    /// Get the mutable field view
    pub fn view_mut(&mut self) -> ArrayViewMut3<f64> {
        self.guard.index_axis_mut(Axis(0), self.field_index)
    }

    /// Get an immutable view
    pub fn view(&self) -> ArrayView3<f64> {
        self.guard.index_axis(Axis(0), self.field_index)
    }
}

// Note: FieldWriteGuard does not implement Deref/DerefMut directly
// because it cannot safely return a &mut Array3<f64> without owning the data.
// Use view_mut() method instead for mutable access.

impl PhysicsState {
    /// Create a new physics state with the given grid
    pub fn new(grid: Grid) -> Self {
        let (nx, ny, nz) = grid.dimensions();
        let fields = Arc::new(RwLock::new(Array4::<f64>::zeros((
            field_indices::TOTAL_FIELDS,
            nx,
            ny,
            nz,
        ))));

        let mut field_names = HashMap::new();
        field_names.insert(field_indices::PRESSURE_IDX, "Pressure".to_string());
        field_names.insert(field_indices::TEMPERATURE_IDX, "Temperature".to_string());
        field_names.insert(
            field_indices::BUBBLE_RADIUS_IDX,
            "Bubble Radius".to_string(),
        );
        field_names.insert(
            field_indices::BUBBLE_VELOCITY_IDX,
            "Bubble Velocity".to_string(),
        );
        field_names.insert(field_indices::DENSITY_IDX, "Density".to_string());
        field_names.insert(field_indices::SOUND_SPEED_IDX, "Sound Speed".to_string());
        field_names.insert(field_indices::VX_IDX, "Velocity X".to_string());
        field_names.insert(field_indices::VY_IDX, "Velocity Y".to_string());
        field_names.insert(field_indices::VZ_IDX, "Velocity Z".to_string());

        let mut field_units = HashMap::new();
        field_units.insert(field_indices::PRESSURE_IDX, "Pa".to_string());
        field_units.insert(field_indices::TEMPERATURE_IDX, "K".to_string());
        field_units.insert(field_indices::BUBBLE_RADIUS_IDX, "m".to_string());
        field_units.insert(field_indices::BUBBLE_VELOCITY_IDX, "m/s".to_string());
        field_units.insert(field_indices::DENSITY_IDX, "kg/m³".to_string());
        field_units.insert(field_indices::SOUND_SPEED_IDX, "m/s".to_string());
        field_units.insert(field_indices::VX_IDX, "m/s".to_string());
        field_units.insert(field_indices::VY_IDX, "m/s".to_string());
        field_units.insert(field_indices::VZ_IDX, "m/s".to_string());

        Self {
            fields,
            grid,
            field_names,
            field_units,
        }
    }

    /// Get a read-only view of a specific field (zero-copy)
    pub fn get_field(&self, field_index: usize) -> KwaversResult<FieldReadGuard<'_>> {
        if field_index >= field_indices::TOTAL_FIELDS {
            return Err(PhysicsError::InvalidFieldIndex(field_index).into());
        }

        let guard = self
            .fields
            .read()
            .map_err(|e| PhysicsError::StateError(format!("Failed to acquire read lock: {}", e)))?;

        Ok(FieldReadGuard::new(guard, field_index))
    }

    /// Get a mutable view of a specific field (zero-copy)
    pub fn get_field_mut(&self, field_index: usize) -> KwaversResult<FieldWriteGuard<'_>> {
        if field_index >= field_indices::TOTAL_FIELDS {
            return Err(PhysicsError::InvalidFieldIndex(field_index).into());
        }

        let guard = self.fields.write().map_err(|e| {
            PhysicsError::StateError(format!("Failed to acquire write lock: {}", e))
        })?;

        Ok(FieldWriteGuard { guard, field_index })
    }

    /// Apply a closure to a field for reading (zero-copy)
    pub fn with_field<F, R>(&self, field_index: usize, f: F) -> KwaversResult<R>
    where
        F: FnOnce(ArrayView3<f64>) -> R,
    {
        if field_index >= field_indices::TOTAL_FIELDS {
            return Err(PhysicsError::InvalidFieldIndex(field_index).into());
        }

        let fields = self
            .fields
            .read()
            .map_err(|e| PhysicsError::StateError(format!("Failed to acquire read lock: {}", e)))?;

        Ok(f(fields.index_axis(Axis(0), field_index)))
    }

    /// Apply a closure to a field for mutation (zero-copy)
    pub fn with_field_mut<F, R>(&self, field_index: usize, f: F) -> KwaversResult<R>
    where
        F: FnOnce(ArrayViewMut3<f64>) -> R,
    {
        if field_index >= field_indices::TOTAL_FIELDS {
            return Err(PhysicsError::InvalidFieldIndex(field_index).into());
        }

        let mut fields = self.fields.write().map_err(|e| {
            PhysicsError::StateError(format!("Failed to acquire write lock: {}", e))
        })?;

        Ok(f(fields.index_axis_mut(Axis(0), field_index)))
    }

    /// Get a cloned copy of a field (allocates memory)
    /// Prefer get_field() for zero-copy access when possible
    pub fn clone_field(&self, field_index: usize) -> KwaversResult<Array3<f64>> {
        self.with_field(field_index, |field| field.to_owned())
    }

    /// Update a specific field with new data
    pub fn update_field(&self, field_index: usize, data: &Array3<f64>) -> KwaversResult<()> {
        if field_index >= field_indices::TOTAL_FIELDS {
            return Err(PhysicsError::InvalidFieldIndex(field_index).into());
        }

        let mut fields = self.fields.write().map_err(|e| {
            PhysicsError::StateError(format!("Failed to acquire write lock: {}", e))
        })?;

        let mut field = fields.index_axis_mut(Axis(0), field_index);
        if data.shape() != field.shape() {
            return Err(PhysicsError::DimensionMismatch.into());
        }
        field.assign(data);
        Ok(())
    }

    /// Get direct access to all fields (for plugin system)
    /// This is more efficient than individual field access when multiple fields are needed
    pub fn with_all_fields_mut<F, R>(&self, f: F) -> KwaversResult<R>
    where
        F: FnOnce(&mut Array4<f64>) -> R,
    {
        let mut fields = self.fields.write().map_err(|e| {
            PhysicsError::StateError(format!("Failed to acquire write lock: {}", e))
        })?;

        Ok(f(&mut fields))
    }

    /// Get all fields
    pub fn get_all_fields(&self) -> KwaversResult<Array4<f64>> {
        let fields = self
            .fields
            .read()
            .map_err(|e| PhysicsError::StateError(format!("Failed to acquire read lock: {}", e)))?;

        Ok(fields.clone())
    }

    /// Get field metadata
    pub fn get_field_name(&self, field_index: usize) -> Option<&String> {
        self.field_names.get(&field_index)
    }

    pub fn get_field_unit(&self, field_index: usize) -> Option<&String> {
        self.field_units.get(&field_index)
    }

    /// Initialize field with a constant value
    pub fn initialize_field(&self, field_index: usize, value: f64) -> KwaversResult<()> {
        self.with_field_mut(field_index, |mut field| {
            field.fill(value);
            Ok(())
        })?
    }

    /// Initialize field with a function
    pub fn initialize_field_with<F>(&self, field_index: usize, init_fn: F) -> KwaversResult<()>
    where
        F: Fn(usize, usize, usize) -> f64,
    {
        self.with_field_mut(field_index, |mut field| {
            let (nx, ny, nz) = (field.shape()[0], field.shape()[1], field.shape()[2]);
            // Use iterator combinators for better performance (CUPID: Composable)
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        field[[i, j, k]] = init_fn(i, j, k);
                    }
                }
            }
            Ok(())
        })?
    }

    /// Get grid reference
    pub fn grid(&self) -> &Grid {
        &self.grid
    }
}

/// Trait for types that provide access to physics state
pub trait HasPhysicsState {
    /// Get reference to the physics state
    fn physics_state(&self) -> &PhysicsState;

    /// Get a specific field by index
    fn get_field(&self, field_index: usize) -> KwaversResult<FieldReadGuard<'_>> {
        self.physics_state().get_field(field_index)
    }

    /// Update a specific field
    fn update_field(&self, field_index: usize, data: &Array3<f64>) -> KwaversResult<()> {
        self.physics_state().update_field(field_index, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physics_state_creation() {
        let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1);
        let state = PhysicsState::new(grid);

        // Test field retrieval
        let pressure = state.get_field(field_indices::PRESSURE_IDX).unwrap();
        assert_eq!(pressure.view().shape(), &[10, 10, 10]);

        // Test field initialization
        state
            .initialize_field(field_indices::TEMPERATURE_IDX, 293.15)
            .unwrap();
        let temp = state.get_field(field_indices::TEMPERATURE_IDX).unwrap();
        assert!((temp.view()[[5, 5, 5]] - 293.15).abs() < 1e-10);
    }

    #[test]
    fn test_field_updates() {
        let grid = Grid::new(5, 5, 5, 0.1, 0.1, 0.1);
        let state = PhysicsState::new(grid);

        // Create test data
        let mut test_data = Array3::zeros((5, 5, 5));
        test_data[[2, 2, 2]] = 100.0;

        // Update field
        state
            .update_field(field_indices::PRESSURE_IDX, &test_data)
            .unwrap();

        // Verify update
        let pressure = state.get_field(field_indices::PRESSURE_IDX).unwrap();
        assert_eq!(pressure.view()[[2, 2, 2]], 100.0);
    }

    #[test]
    fn test_zero_copy_access() {
        let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1);
        let state = PhysicsState::new(grid);

        // Test zero-copy read
        state
            .with_field(field_indices::PRESSURE_IDX, |field| {
                // This should not allocate
                assert_eq!(field.shape(), &[10, 10, 10]);
            })
            .unwrap();

        // Test zero-copy write
        state
            .with_field_mut(field_indices::TEMPERATURE_IDX, |mut field| {
                field[[5, 5, 5]] = 300.0;
            })
            .unwrap();

        // Verify the write
        state
            .with_field(field_indices::TEMPERATURE_IDX, |field| {
                assert_eq!(field[[5, 5, 5]], 300.0);
            })
            .unwrap();
    }

    #[test]
    fn test_field_guard_deref() {
        let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1);
        let state = PhysicsState::new(grid);

        // Initialize field
        state
            .initialize_field(field_indices::PRESSURE_IDX, 101325.0)
            .unwrap();

        // Test read guard deref
        let guard = state.get_field(field_indices::PRESSURE_IDX).unwrap();
        assert_eq!(guard[[0, 0, 0]], 101325.0);

        // Test write guard deref
        let mut guard = state.get_field_mut(field_indices::TEMPERATURE_IDX).unwrap();
        let mut view = guard.view_mut();
        view[[0, 0, 0]] = 273.15;
        drop(view);
        drop(guard);

        // Verify write
        let guard = state.get_field(field_indices::TEMPERATURE_IDX).unwrap();
        assert_eq!(guard[[0, 0, 0]], 273.15);
    }
}
