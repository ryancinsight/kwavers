//! Advanced GPU Memory Management with Multi-GPU Support
//!
//! Provides unified memory, memory pooling, streaming transfers, and compression
//! for optimal multi-GPU performance.

mod compression;
mod pool;
mod streaming;
#[cfg(test)]
mod tests;

pub use compression::{CompressedBlock, MemoryCompression};
pub use pool::{GpuMemoryPool, GpuMemoryPoolStats, MemoryBlock, MemoryHandle};
pub use streaming::{StreamingTransferManager, TransferStream, UnifiedMemoryRegion};

use crate::core::error::KwaversResult;
use std::collections::HashMap;

/// Memory pool types for different usage patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuMemoryPoolType {
    /// Temporary buffers for intermediate computations
    Temporary,
    /// Persistent buffers for long-term storage
    Persistent,
    /// Collocation points for PINN training
    Collocation,
    /// Domain decomposition boundary data
    Boundary,
    /// FFT working memory
    FFT,
}

/// Unified memory manager for multi-GPU systems
#[derive(Debug)]
pub struct UnifiedMemoryManager {
    pools: HashMap<usize, HashMap<GpuMemoryPoolType, GpuMemoryPool>>,
    unified_regions: Vec<UnifiedMemoryRegion>,
    compression: MemoryCompression,
    streaming: StreamingTransferManager,
}

impl UnifiedMemoryManager {
    /// Create new unified memory manager
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
            unified_regions: Vec::new(),
            compression: MemoryCompression::new(),
            streaming: StreamingTransferManager::new(),
        }
    }

    /// Allocate memory in specified pool on specific GPU
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn allocate(
        &mut self,
        gpu_id: usize,
        pool_type: GpuMemoryPoolType,
        size: usize,
    ) -> KwaversResult<MemoryHandle> {
        let pools = self.pools.entry(gpu_id).or_default();
        let pool = pools
            .entry(pool_type)
            .or_insert_with(|| GpuMemoryPool::new(pool_type));
        pool.allocate(gpu_id, size)
    }

    /// Deallocate memory
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn deallocate(&mut self, handle: MemoryHandle) -> KwaversResult<()> {
        if let Some(pools) = self.pools.get_mut(&handle.gpu_id) {
            if let Some(pool) = pools.get_mut(&handle.pool_type) {
                pool.deallocate(handle)
            } else {
                Err("Memory pool not found".into())
            }
        } else {
            Err("GPU device not found".into())
        }
    }

    /// Transfer data between GPUs with streaming optimization
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn transfer(
        &mut self,
        src: &MemoryHandle,
        dst: &MemoryHandle,
        size: usize,
    ) -> KwaversResult<()> {
        if let Some(region) = self.find_unified_region(src, dst) {
            self.streaming.unified_transfer(src, dst, size, &region)
        } else {
            self.streaming.streaming_transfer(src, dst, size)
        }
    }

    /// Compress memory region for storage optimization
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn compress(&mut self, handle: &MemoryHandle) -> KwaversResult<f64> {
        self.compression.compress(handle)
    }

    /// Decompress memory region
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn decompress(&mut self, handle: &MemoryHandle) -> KwaversResult<()> {
        self.compression.decompress(handle)
    }

    /// Get memory statistics across all GPUs
    pub fn statistics(&self) -> GpuMemoryPoolStats {
        let mut total_stats = GpuMemoryPoolStats::default();
        for pools in self.pools.values() {
            for pool in pools.values() {
                let pool_stats = pool.statistics();
                total_stats.allocated_bytes += pool_stats.allocated_bytes;
                total_stats.peak_bytes += pool_stats.peak_bytes;
                total_stats.transfer_count += pool_stats.transfer_count;
                total_stats.transfer_bytes += pool_stats.transfer_bytes;
            }
        }
        total_stats
    }

    fn find_unified_region(
        &self,
        src: &MemoryHandle,
        dst: &MemoryHandle,
    ) -> Option<UnifiedMemoryRegion> {
        self.unified_regions
            .iter()
            .find(|r| r.contains_gpu(src.gpu_id) && r.contains_gpu(dst.gpu_id))
            .cloned()
    }
}

impl Default for UnifiedMemoryManager {
    fn default() -> Self {
        Self::new()
    }
}
