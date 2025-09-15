//! **DEPRECATED - MEMORY SAFETY VIOLATION**
//! 
//! This module contained unsafe code with RefCell lifetime violations.
//! Removed for production safety. Use proper Arc<RwLock<>> patterns instead.
//!
//! **CRITICAL SAFETY ISSUE**: The original implementation created ArrayView3
//! from raw pointers within RefCell borrow scope, causing use-after-free
//! when the view outlived the borrow. This is a production-blocking bug.

use crate::grid::Grid;
use crate::KwaversResult;
use ndarray::{Array3, Array4, ArrayView3, ArrayViewMut3, Axis};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// **SAFE** physics state using proper synchronization primitives
/// 
/// Replaces the unsafe thread-local implementation with Arc<RwLock<>>
/// for memory-safe concurrent access without RefCell lifetime violations.
#[derive(Clone)]
pub struct SafePhysicsState {
    inner: Arc<RwLock<PhysicsStateData>>,
}

/// Internal physics state data protected by RwLock
struct PhysicsStateData {
    grid: Grid,
    fields: Array4<f64>,
    field_names: HashMap<usize, String>,
    field_units: HashMap<usize, String>,
    generation: u64,
}

impl SafePhysicsState {
    /// Create new physics state with memory-safe implementation
    pub fn new(grid: Grid, num_fields: usize) -> Self {
        let shape = (num_fields, grid.nx, grid.ny, grid.nz);
        
        Self {
            inner: Arc::new(RwLock::new(PhysicsStateData {
                grid,
                fields: Array4::zeros(shape),
                field_names: HashMap::new(),
                field_units: HashMap::new(),
                generation: 0,
            })),
        }
    }
    
    /// Get read-only access to a field with proper lifetime management
    /// 
    /// Returns a copy instead of a view to avoid lifetime complications.
    /// For performance-critical code, use with_field_view() with a closure.
    pub fn get_field_copy(&self, field_index: usize) -> KwaversResult<Array3<f64>> {
        let state = self.inner.read()
            .map_err(|_| "RwLock poisoned during read")?;
            
        if field_index >= state.fields.shape()[0] {
            return Err(format!("Field index {} out of bounds", field_index).into());
        }
        
        Ok(state.fields.index_axis(Axis(0), field_index).to_owned())
    }
    
    /// Access field with a closure to avoid lifetime issues (zero-copy)
    pub fn with_field_view<T, F>(&self, field_index: usize, f: F) -> KwaversResult<T>
    where
        F: FnOnce(ArrayView3<f64>) -> T,
    {
        let state = self.inner.read()
            .map_err(|_| "RwLock poisoned during read")?;
            
        if field_index >= state.fields.shape()[0] {
            return Err(format!("Field index {} out of bounds", field_index).into());
        }
        
        let view = state.fields.index_axis(Axis(0), field_index);
        Ok(f(view))
    }
    
    /// Update a field safely
    pub fn update_field(&self, field_index: usize, data: Array3<f64>) -> KwaversResult<()> {
        let mut state = self.inner.write()
            .map_err(|_| "RwLock poisoned during write")?;
            
        if field_index >= state.fields.shape()[0] {
            return Err(format!("Field index {} out of bounds", field_index).into());
        }
        
        state.fields.index_axis_mut(Axis(0), field_index).assign(&data);
        state.generation += 1;
        
        Ok(())
    }
    
    /// Get current generation (for cache validation)
    pub fn generation(&self) -> KwaversResult<u64> {
        let state = self.inner.read()
            .map_err(|_| "RwLock poisoned during read")?;
        Ok(state.generation)
    }
}
    
    /// Get mutable access to a field (requires synchronization)
    pub fn get_field_mut(&self, field_index: usize) -> KwaversResult<ArrayViewMut3<f64>> {
        self.sync_local_state();
        
        LOCAL_STATE.with(|state| {
            let mut state_ref = state.borrow_mut();
            let local = state_ref.as_mut()
                .ok_or_else(|| "Local state not initialized".to_string())?;
            
            if field_index >= local.fields.shape()[0] {
                return Err(format!("Field index {} out of bounds", field_index).into());
            }
            
            // This is safe because we have exclusive access through RefCell
            unsafe {
                let ptr = local.fields.as_mut_ptr();
                let shape = local.fields.shape();
                let strides = local.fields.strides();
                
                // Calculate offset for the field
                let field_offset = field_index * (shape[1] * shape[2] * shape[3]);
                let field_ptr = ptr.add(field_offset);
                
                // Create mutable view from raw parts
                let field_shape = [shape[1], shape[2], shape[3]];
                let field_strides = [strides[1], strides[2], strides[3]];
                
                Ok(ArrayViewMut3::from_shape_ptr(
                    field_shape.strides(field_strides),
                    field_ptr
                ))
            }
/// Type alias for backward compatibility during migration
/// TODO: Remove once all references are updated to SafePhysicsState
#[deprecated(since = "2.14.0", note = "Use SafePhysicsState - the original implementation had memory safety violations")]
pub type LockFreePhysicsState = SafePhysicsState;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lockfree_state() {
        let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1).unwrap();
        let state = LockFreePhysicsState::new(grid, 3);
        
        // Test field access
        let field = state.get_field(0).unwrap();
        assert_eq!(field.shape(), &[10, 10, 10]);
        
        // Test field update
        let mut data = Array3::ones((10, 10, 10));
        state.update_field(0, data.clone()).unwrap();
        
        // Verify update
        let field = state.get_field(0).unwrap();
        assert_eq!(field[[5, 5, 5]], 1.0);
    }
}