//! Storage backends for recording

pub mod file;
pub mod memory;

pub use file::FileStorage;
pub use memory::MemoryStorage;

use crate::domain::core::error::KwaversResult;
use ndarray::Array3;

/// Storage backend trait
pub trait StorageBackend: Send + Sync {
    /// Initialize storage
    fn initialize(&mut self, shape: (usize, usize, usize)) -> KwaversResult<()>;

    /// Store field data
    fn store_field(&mut self, name: &str, field: &Array3<f64>, step: usize) -> KwaversResult<()>;

    /// Finalize storage
    fn finalize(&mut self) -> KwaversResult<()>;
}
