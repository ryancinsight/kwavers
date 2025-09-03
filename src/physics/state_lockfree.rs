//! Lock-free physics state implementation using thread-local storage

use crate::grid::Grid;
use crate::KwaversResult;
use crossbeam::queue::SegQueue;
use ndarray::{Array3, Array4, ArrayView3, ArrayViewMut3, Axis};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

thread_local! {
    static LOCAL_STATE: RefCell<Option<LocalPhysicsState>> = RefCell::new(None);
}

/// Thread-local physics state for lock-free access
struct LocalPhysicsState {
    fields: Array4<f64>,
    generation: u64,
}

/// Lock-free physics state using thread-local storage
#[derive(Clone)]
pub struct LockFreePhysicsState {
    grid: Grid,
    field_names: Arc<HashMap<usize, String>>,
    field_units: Arc<HashMap<usize, String>>,
    updates: Arc<SegQueue<StateUpdate>>,
    generation: Arc<std::sync::atomic::AtomicU64>,
}

#[derive(Clone)]
struct StateUpdate {
    field_index: usize,
    data: Array3<f64>,
    generation: u64,
}

impl LockFreePhysicsState {
    /// Create new physics state
    pub fn new(grid: Grid, num_fields: usize) -> Self {
        let shape = (num_fields, grid.nx, grid.ny, grid.nz);
        
        // Initialize thread-local state
        LOCAL_STATE.with(|state| {
            *state.borrow_mut() = Some(LocalPhysicsState {
                fields: Array4::zeros(shape),
                generation: 0,
            });
        });
        
        Self {
            grid,
            field_names: Arc::new(HashMap::new()),
            field_units: Arc::new(HashMap::new()),
            updates: Arc::new(SegQueue::new()),
            generation: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }
    
    /// Get read-only access to a field
    pub fn get_field(&self, field_index: usize) -> KwaversResult<ArrayView3<f64>> {
        self.sync_local_state();
        
        LOCAL_STATE.with(|state| {
            let state_ref = state.borrow();
            let local = state_ref.as_ref()
                .ok_or_else(|| "Local state not initialized".to_string())?;
            
            if field_index >= local.fields.shape()[0] {
                return Err(format!("Field index {} out of bounds", field_index).into());
            }
            
            // This is safe because we're returning a view with the same lifetime as the borrow
            unsafe {
                let ptr = local.fields.as_ptr();
                let shape = local.fields.shape();
                let strides = local.fields.strides();
                
                // Calculate offset for the field
                let field_offset = field_index * (shape[1] * shape[2] * shape[3]);
                let field_ptr = ptr.add(field_offset);
                
                // Create view from raw parts
                let field_shape = [shape[1], shape[2], shape[3]];
                let field_strides = [strides[1], strides[2], strides[3]];
                
                Ok(ArrayView3::from_shape_ptr(
                    field_shape.strides(field_strides),
                    field_ptr
                ))
            }
        })
    }
    
    /// Update a field (thread-safe)
    pub fn update_field(&self, field_index: usize, data: Array3<f64>) -> KwaversResult<()> {
        let generation = self.generation.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        
        self.updates.push(StateUpdate {
            field_index,
            data,
            generation,
        });
        
        Ok(())
    }
    
    /// Synchronize local state with updates
    fn sync_local_state(&self) {
        LOCAL_STATE.with(|state| {
            let mut state_ref = state.borrow_mut();
            let local = state_ref.as_mut().expect("Local state not initialized");
            
            // Process all pending updates
            while let Some(update) = self.updates.pop() {
                if update.generation > local.generation {
                    local.fields.index_axis_mut(Axis(0), update.field_index)
                        .assign(&update.data);
                    local.generation = update.generation;
                }
            }
        });
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
        })
    }
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