//! Centralized physics state management following SOLID principles
//!
//! This module provides a single source of truth for physics field states,
//! eliminating the need for dummy fields scattered across implementations.

use crate::error::{KwaversResult, PhysicsError};
use crate::grid::Grid;
use ndarray::{Array3, Array4, ArrayView3, ArrayViewMut3, Axis};
use std::collections::HashMap;

// Use the global field_indices module for consistency
pub use crate::physics::field_indices;

/// Physics state container - Single Source of Truth for all field data
///
/// This struct owns the field data directly, avoiding unnecessary Arc<RwLock>
/// indirection. For concurrent access, wrap the entire PhysicsState in Arc<RwLock>
/// at the application level if needed.
#[derive(Debug)]
pub struct PhysicsState {
    /// Main 4D field array containing all physics quantities
    fields: Array4<f64>,

    /// Grid dimensions
    grid: Grid,

    /// Field metadata
    field_names: HashMap<usize, String>,
    field_units: HashMap<usize, String>,
}

/// Direct field view for zero-copy read access
///
/// Since PhysicsState now owns data directly, we can return simple borrows
/// instead of complex guard types that clone data unnecessarily.
pub type FieldView<'a> = ArrayView3<'a, f64>;

/// Direct mutable field view for zero-copy write access
pub type FieldViewMut<'a> = ArrayViewMut3<'a, f64>;

impl PhysicsState {
    /// Create a new physics state with the given grid
    pub fn new(grid: Grid) -> Self {
        let (nx, ny, nz) = grid.dimensions();
        let fields = Array4::<f64>::zeros((field_indices::TOTAL_FIELDS, nx, ny, nz));

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
    pub fn get_field(&self, field_index: usize) -> KwaversResult<FieldView<'_>> {
        if field_index >= field_indices::TOTAL_FIELDS {
            return Err(PhysicsError::InvalidFieldIndex(field_index).into());
        }

        Ok(self.fields.index_axis(Axis(0), field_index))
    }

    /// Get a mutable view of a specific field (zero-copy)
    pub fn get_field_mut(&mut self, field_index: usize) -> KwaversResult<FieldViewMut<'_>> {
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
    /// This is more efficient than individual field access when multiple fields are needed
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

    /// Get field metadata
    pub fn get_field_name(&self, field_index: usize) -> Option<&String> {
        self.field_names.get(&field_index)
    }

    pub fn get_field_unit(&self, field_index: usize) -> Option<&String> {
        self.field_units.get(&field_index)
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
            // Use parallel iteration for better performance when available
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physics_state_creation() {
        // RIGOROUS VALIDATION: Physics state creation and field access must be fast and correct
        println!("Creating grid...");
        let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1).unwrap();
        println!("Creating PhysicsState...");
        let state = PhysicsState::new(grid);
        println!("PhysicsState created successfully");

        // EXACT ASSERTION: Field retrieval must work correctly
        println!("Getting pressure field...");
        let pressure = state.get_field(field_indices::PRESSURE_IDX).unwrap();
        println!("Got pressure field with shape: {:?}", pressure.shape());

        // EXACT VALIDATION: Dimensions must match grid exactly
        assert_eq!(
            pressure.shape(),
            &[10, 10, 10],
            "Pressure field dimensions don't match grid"
        );

        // EXACT ASSERTION: Field initialization must be precise
        let mut state = state; // Make mutable for initialization
        state
            .initialize_field(field_indices::TEMPERATURE_IDX, 293.15)
            .unwrap();
        let temp = state.get_field(field_indices::TEMPERATURE_IDX).unwrap();

        // EXACT VALIDATION: Initialization value must be preserved exactly
        assert!(
            (temp[[5, 5, 5]] - 293.15).abs() < f64::EPSILON,
            "Temperature initialization failed: expected 293.15, got {}",
            temp[[5, 5, 5]]
        );
    }

    #[test]
    fn test_field_updates() {
        let grid = Grid::new(5, 5, 5, 0.1, 0.1, 0.1).unwrap();
        let mut state = PhysicsState::new(grid);

        // Create test data
        let mut test_data = Array3::zeros((5, 5, 5));
        test_data[[2, 2, 2]] = 100.0;

        // Update field
        state
            .update_field(field_indices::PRESSURE_IDX, &test_data)
            .unwrap();

        // Verify update
        let pressure = state.get_field(field_indices::PRESSURE_IDX).unwrap();
        assert_eq!(pressure[[2, 2, 2]], 100.0);
    }

    #[test]
    fn test_zero_copy_access() {
        let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1).unwrap();
        let mut state = PhysicsState::new(grid);

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
        // RIGOROUS VALIDATION: Field access must be deadlock-free and correct
        let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1).unwrap();
        let mut state = PhysicsState::new(grid);

        // EXACT ASSERTION: Field initialization must work precisely
        state
            .initialize_field(field_indices::PRESSURE_IDX, 101325.0)
            .unwrap();

        // EXACT VALIDATION: Direct field access must return exact values
        let pressure = state.get_field(field_indices::PRESSURE_IDX).unwrap();
        assert_eq!(
            pressure[[0, 0, 0]],
            101325.0,
            "Pressure field initialization failed"
        );

        // EXACT ASSERTION: Mutable field access must work without deadlocks
        {
            let mut temp = state.get_field_mut(field_indices::TEMPERATURE_IDX).unwrap();
            temp[[0, 0, 0]] = 273.15;
        } // Scope ensures mutable borrow is dropped

        // EXACT VALIDATION: Write operations must persist correctly
        let temp = state.get_field(field_indices::TEMPERATURE_IDX).unwrap();
        assert_eq!(
            temp[[0, 0, 0]],
            273.15,
            "Temperature field write operation failed"
        );
    }
}
