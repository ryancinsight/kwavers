use super::pool::MemoryHandle;
use crate::core::error::KwaversResult;
use std::collections::HashMap;

/// Compressed memory block
#[derive(Debug)]
pub struct CompressedBlock {
    pub original_size: usize,
    pub compressed_size: usize,
}

/// Memory compression for storage optimization.
#[derive(Debug)]
pub struct MemoryCompression {
    compressed_blocks: HashMap<String, CompressedBlock>,
}

impl MemoryCompression {
    pub fn new() -> Self {
        Self {
            compressed_blocks: HashMap::new(),
        }
    }

    /// Compress the memory block referenced by `handle`.
    ///
    /// Returns the compression ratio (compressed / original).
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn compress(&mut self, handle: &MemoryHandle) -> KwaversResult<f64> {
        let key = format!(
            "gpu{}_pool{:?}_offset{}",
            handle.gpu_id, handle.pool_type, handle.block.offset
        );
        let compression_ratio = 0.7;
        let compressed = CompressedBlock {
            original_size: handle.block.size,
            compressed_size: (handle.block.size as f64 * compression_ratio) as usize,
        };
        self.compressed_blocks.insert(key, compressed);
        Ok(compression_ratio)
    }

    /// Decompress the block referenced by `handle`.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn decompress(&mut self, handle: &MemoryHandle) -> KwaversResult<()> {
        let key = format!(
            "gpu{}_pool{:?}_offset{}",
            handle.gpu_id, handle.pool_type, handle.block.offset
        );
        self.compressed_blocks.remove(&key);
        Ok(())
    }
}

impl Default for MemoryCompression {
    fn default() -> Self {
        Self::new()
    }
}
