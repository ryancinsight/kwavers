//! Memory-based storage backend

use super::StorageBackend;
use crate::KwaversResult;
use ndarray::Array3;
use std::collections::HashMap;

/// Memory storage backend
pub struct MemoryStorage {
    data: HashMap<String, Vec<Array3<f64>>>,
    shape: Option<(usize, usize, usize)>,
}

impl MemoryStorage {
    /// Create memory storage
    pub fn create() -> Self {
        Self {
            data: HashMap::new(),
            shape: None,
        }
    }

    /// Get stored data
    pub fn get_data(&self, name: &str) -> Option<&Vec<Array3<f64>>> {
        self.data.get(name)
    }
}

impl StorageBackend for MemoryStorage {
    fn initialize(&mut self, shape: (usize, usize, usize)) -> KwaversResult<()> {
        self.shape = Some(shape);
        Ok(())
    }

    fn store_field(&mut self, name: &str, field: &Array3<f64>, _step: usize) -> KwaversResult<()> {
        self.data
            .entry(name.to_string())
            .or_default()
            .push(field.clone());
        Ok(())
    }

    fn finalize(&mut self) -> KwaversResult<()> {
        // No cleanup needed for memory storage
        Ok(())
    }
}
