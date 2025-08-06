//! Centralized physics state management following SOLID principles
//! 
//! This module provides a single source of truth for physics field states,
//! eliminating the need for dummy fields scattered across implementations.

use crate::error::{KwaversResult, PhysicsError};
use crate::grid::Grid;
use ndarray::{Array3, Array4, Axis};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Field indices for the 4D field array
pub mod field_indices {
    pub const PRESSURE: usize = 0;
    pub const TEMPERATURE: usize = 1;
    pub const BUBBLE_RADIUS: usize = 2;
    pub const BUBBLE_VELOCITY: usize = 3;
    pub const DENSITY: usize = 4;
    pub const SOUND_SPEED: usize = 5;
    pub const VELOCITY_X: usize = 6;
    pub const VELOCITY_Y: usize = 7;
    pub const VELOCITY_Z: usize = 8;
    pub const STRESS_XX: usize = 9;
    pub const STRESS_YY: usize = 10;
    pub const STRESS_ZZ: usize = 11;
    pub const STRESS_XY: usize = 12;
    pub const STRESS_XZ: usize = 13;
    pub const STRESS_YZ: usize = 14;
    pub const LIGHT_FLUENCE: usize = 15;
    pub const CHEMICAL_CONCENTRATION: usize = 16;
    
    pub const TOTAL_FIELDS: usize = 17;
}

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

impl PhysicsState {
    /// Create a new physics state with the given grid
    pub fn new(grid: &Grid) -> Self {
        let fields = Array4::zeros((
            field_indices::TOTAL_FIELDS,
            grid.nx,
            grid.ny,
            grid.nz,
        ));
        
        let mut field_names = HashMap::new();
        field_names.insert(field_indices::PRESSURE, "Pressure".to_string());
        field_names.insert(field_indices::TEMPERATURE, "Temperature".to_string());
        field_names.insert(field_indices::BUBBLE_RADIUS, "Bubble Radius".to_string());
        field_names.insert(field_indices::BUBBLE_VELOCITY, "Bubble Velocity".to_string());
        field_names.insert(field_indices::DENSITY, "Density".to_string());
        field_names.insert(field_indices::SOUND_SPEED, "Sound Speed".to_string());
        field_names.insert(field_indices::VELOCITY_X, "Velocity X".to_string());
        field_names.insert(field_indices::VELOCITY_Y, "Velocity Y".to_string());
        field_names.insert(field_indices::VELOCITY_Z, "Velocity Z".to_string());
        field_names.insert(field_indices::STRESS_XX, "Stress XX".to_string());
        field_names.insert(field_indices::STRESS_YY, "Stress YY".to_string());
        field_names.insert(field_indices::STRESS_ZZ, "Stress ZZ".to_string());
        field_names.insert(field_indices::STRESS_XY, "Stress XY".to_string());
        field_names.insert(field_indices::STRESS_XZ, "Stress XZ".to_string());
        field_names.insert(field_indices::STRESS_YZ, "Stress YZ".to_string());
        field_names.insert(field_indices::LIGHT_FLUENCE, "Light Fluence".to_string());
        field_names.insert(field_indices::CHEMICAL_CONCENTRATION, "Chemical Concentration".to_string());
        
        let mut field_units = HashMap::new();
        field_units.insert(field_indices::PRESSURE, "Pa".to_string());
        field_units.insert(field_indices::TEMPERATURE, "K".to_string());
        field_units.insert(field_indices::BUBBLE_RADIUS, "m".to_string());
        field_units.insert(field_indices::BUBBLE_VELOCITY, "m/s".to_string());
        field_units.insert(field_indices::DENSITY, "kg/m³".to_string());
        field_units.insert(field_indices::SOUND_SPEED, "m/s".to_string());
        field_units.insert(field_indices::VELOCITY_X, "m/s".to_string());
        field_units.insert(field_indices::VELOCITY_Y, "m/s".to_string());
        field_units.insert(field_indices::VELOCITY_Z, "m/s".to_string());
        field_units.insert(field_indices::STRESS_XX, "Pa".to_string());
        field_units.insert(field_indices::STRESS_YY, "Pa".to_string());
        field_units.insert(field_indices::STRESS_ZZ, "Pa".to_string());
        field_units.insert(field_indices::STRESS_XY, "Pa".to_string());
        field_units.insert(field_indices::STRESS_XZ, "Pa".to_string());
        field_units.insert(field_indices::STRESS_YZ, "Pa".to_string());
        field_units.insert(field_indices::LIGHT_FLUENCE, "J/m²".to_string());
        field_units.insert(field_indices::CHEMICAL_CONCENTRATION, "mol/m³".to_string());
        
        Self {
            fields: Arc::new(RwLock::new(fields)),
            grid: grid.clone(),
            field_names,
            field_units,
        }
    }
    
    /// Get a read-only view of a specific field
    pub fn get_field(&self, field_index: usize) -> KwaversResult<Array3<f64>> {
        let fields = self.fields.read().map_err(|e| 
            PhysicsError::StateError(format!("Failed to acquire read lock: {}", e))
        )?;
        
        if field_index >= field_indices::TOTAL_FIELDS {
            return Err(PhysicsError::InvalidFieldIndex(field_index).into());
        }
        
        Ok(fields.index_axis(Axis(0), field_index).to_owned())
    }
    
    /// Get a mutable view of a specific field
    pub fn get_field_mut(&self, field_index: usize) -> KwaversResult<Array3<f64>> {
        let mut fields = self.fields.write().map_err(|e| 
            PhysicsError::StateError(format!("Failed to acquire write lock: {}", e))
        )?;
        
        if field_index >= field_indices::TOTAL_FIELDS {
            return Err(PhysicsError::InvalidFieldIndex(field_index).into());
        }
        
        Ok(fields.index_axis_mut(Axis(0), field_index).to_owned())
    }
    
    /// Update a specific field with new data
    pub fn update_field(&self, field_index: usize, data: &Array3<f64>) -> KwaversResult<()> {
        let mut fields = self.fields.write().map_err(|e| 
            PhysicsError::StateError(format!("Failed to acquire write lock: {}", e))
        )?;
        
        if field_index >= field_indices::TOTAL_FIELDS {
            return Err(PhysicsError::InvalidFieldIndex(field_index).into());
        }
        
        if data.shape() != [self.grid.nx, self.grid.ny, self.grid.nz] {
            return Err(PhysicsError::DimensionMismatch.into());
        }
        
        fields.index_axis_mut(Axis(0), field_index).assign(data);
        Ok(())
    }
    
    /// Get all fields
    pub fn get_all_fields(&self) -> KwaversResult<Array4<f64>> {
        let fields = self.fields.read().map_err(|e| 
            PhysicsError::StateError(format!("Failed to acquire read lock: {}", e))
        )?;
        
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
        let mut fields = self.fields.write().map_err(|e| 
            PhysicsError::StateError(format!("Failed to acquire write lock: {}", e))
        )?;
        
        if field_index >= field_indices::TOTAL_FIELDS {
            return Err(PhysicsError::InvalidFieldIndex(field_index).into());
        }
        
        fields.index_axis_mut(Axis(0), field_index).fill(value);
        Ok(())
    }
    
    /// Initialize field with a function
    pub fn initialize_field_with<F>(&self, field_index: usize, init_fn: F) -> KwaversResult<()>
    where
        F: Fn(usize, usize, usize) -> f64,
    {
        let mut fields = self.fields.write().map_err(|e| 
            PhysicsError::StateError(format!("Failed to acquire write lock: {}", e))
        )?;
        
        if field_index >= field_indices::TOTAL_FIELDS {
            return Err(PhysicsError::InvalidFieldIndex(field_index).into());
        }
        
        let mut field = fields.index_axis_mut(Axis(0), field_index);
        for i in 0..self.grid.nx {
            for j in 0..self.grid.ny {
                for k in 0..self.grid.nz {
                    field[[i, j, k]] = init_fn(i, j, k);
                }
            }
        }
        
        Ok(())
    }
}

/// Field accessor trait for components that need field access
pub trait FieldAccessor {
    /// Get the physics state
    fn physics_state(&self) -> &PhysicsState;
    
    /// Get a specific field by index
    fn get_field(&self, field_index: usize) -> KwaversResult<Array3<f64>> {
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
        let state = PhysicsState::new(&grid);
        
        // Test field retrieval
        let pressure = state.get_field(field_indices::PRESSURE).unwrap();
        assert_eq!(pressure.shape(), &[10, 10, 10]);
        
        // Test field initialization
        state.initialize_field(field_indices::TEMPERATURE, 293.15).unwrap();
        let temp = state.get_field(field_indices::TEMPERATURE).unwrap();
        assert!((temp[[5, 5, 5]] - 293.15).abs() < 1e-10);
    }
    
    #[test]
    fn test_field_updates() {
        let grid = Grid::new(5, 5, 5, 0.1, 0.1, 0.1);
        let state = PhysicsState::new(&grid);
        
        // Create test data
        let mut test_data = Array3::zeros((5, 5, 5));
        test_data[[2, 2, 2]] = 100.0;
        
        // Update field
        state.update_field(field_indices::PRESSURE, &test_data).unwrap();
        
        // Verify update
        let pressure = state.get_field(field_indices::PRESSURE).unwrap();
        assert_eq!(pressure[[2, 2, 2]], 100.0);
    }
}