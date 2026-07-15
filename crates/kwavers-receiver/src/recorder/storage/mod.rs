//! Storage backends for recording

pub mod file;
pub mod memory;

pub use file::FileStorage;
pub use memory::MemoryStorage;

use kwavers_core::error::KwaversResult;
use leto::Array3;

/// Storage backend trait
pub trait StorageBackend: Send + Sync {
    /// Initialize storage
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn initialize(&mut self, shape: (usize, usize, usize)) -> KwaversResult<()>;

    /// Store field data
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn store_field(&mut self, name: &str, field: &Array3<f64>, step: usize) -> KwaversResult<()>;

    /// Finalize storage
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn finalize(&mut self) -> KwaversResult<()>;
}
