//! # GPU Memory Management
//!
//! This module provides GPU memory management with allocation strategies,
//! transfer, and caching strategies. Implements Phase 10 performance targets
//! with memory pool management and asynchronous operations.

mod allocation;
mod buffer;
mod cache;
mod pool;
mod transfer;

pub use allocation::{AllocationStrategy, Allocator};
pub use buffer::{BufferDescriptor, GpuBuffer};
pub use cache::{CacheStrategy, MemoryCache};
pub use pool::{MemoryPool, PoolConfig};
pub use transfer::{TransferManager, TransferMode};

use crate::error::KwaversResult;
use crate::gpu::GpuBackend;
use std::sync::Arc;

/// GPU memory manager coordinating all memory operations
#[derive(Debug)]
pub struct GpuMemoryManager {
    allocator: Allocator,
    pool: MemoryPool,
    cache: MemoryCache,
    transfer: TransferManager,
    backend: Arc<dyn GpuBackend>,
}

impl GpuMemoryManager {
    /// Create new GPU memory manager
    pub fn new(backend: Arc<dyn GpuBackend>, strategy: AllocationStrategy) -> KwaversResult<Self> {
        let allocator = Allocator::new(strategy);
        let pool = MemoryPool::new(PoolConfig::default())?;
        let cache = MemoryCache::new(CacheStrategy::Lru);
        let transfer = TransferManager::new(backend.clone());

        Ok(Self {
            allocator,
            pool,
            cache,
            transfer,
            backend,
        })
    }

    /// Allocate GPU memory
    pub fn allocate(&mut self, size_bytes: usize) -> KwaversResult<GpuBuffer> {
        // Try cache first
        if let Some(buffer) = self.cache.get(size_bytes) {
            return Ok(buffer);
        }

        // Try pool
        if let Ok(buffer) = self.pool.allocate(size_bytes) {
            return Ok(buffer);
        }

        // Fall back to allocator
        self.allocator.allocate(size_bytes, &self.backend)
    }

    /// Free GPU memory
    pub fn free(&mut self, buffer: GpuBuffer) -> KwaversResult<()> {
        // Return to pool if possible
        if self.pool.can_recycle(&buffer) {
            return self.pool.recycle(buffer);
        }

        // Otherwise free directly
        self.allocator.free(buffer, &self.backend)
    }

    /// Transfer data to GPU
    pub fn upload(&mut self, host_data: &[u8], buffer: &mut GpuBuffer) -> KwaversResult<()> {
        self.transfer.upload(host_data, buffer)
    }

    /// Transfer data from GPU
    pub fn download(&mut self, buffer: &GpuBuffer, host_data: &mut [u8]) -> KwaversResult<()> {
        self.transfer.download(buffer, host_data)
    }

    /// Get memory statistics
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            allocated_bytes: self.allocator.allocated_bytes(),
            cached_bytes: self.cache.size_bytes(),
            pool_bytes: self.pool.size_bytes(),
            peak_bytes: self.allocator.peak_bytes(),
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub allocated_bytes: usize,
    pub cached_bytes: usize,
    pub pool_bytes: usize,
    pub peak_bytes: usize,
}
