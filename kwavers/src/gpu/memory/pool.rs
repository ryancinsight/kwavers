use super::GpuMemoryPoolType;
use crate::core::error::KwaversResult;

/// Memory handle for allocated blocks
#[derive(Debug, Clone)]
pub struct MemoryHandle {
    pub gpu_id: usize,
    pub pool_type: GpuMemoryPoolType,
    pub block: MemoryBlock,
}

/// Memory block information
#[derive(Debug, Clone)]
pub struct MemoryBlock {
    pub offset: usize,
    pub size: usize,
    pub compressed: bool,
}

/// Memory statistics
#[derive(Debug, Default, Clone)]
pub struct GpuMemoryPoolStats {
    pub allocated_bytes: usize,
    pub peak_bytes: usize,
    pub transfer_count: usize,
    pub transfer_bytes: usize,
    pub compression_ratio: f64,
}

/// Memory pool for efficient allocation
#[derive(Debug)]
pub struct GpuMemoryPool {
    pub(super) pool_type: GpuMemoryPoolType,
    pub(super) allocations: Vec<MemoryBlock>,
    pub(super) total_allocated: usize,
    pub(super) peak_allocated: usize,
}

impl GpuMemoryPool {
    /// Create a new memory pool.
    pub fn new(pool_type: GpuMemoryPoolType) -> Self {
        Self {
            pool_type,
            allocations: Vec::new(),
            total_allocated: 0,
            peak_allocated: 0,
        }
    }

    /// Allocate a block of `size` bytes on `gpu_id`.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn allocate(&mut self, gpu_id: usize, size: usize) -> KwaversResult<MemoryHandle> {
        let offset = self.total_allocated;
        self.total_allocated = self.total_allocated.saturating_add(size);
        self.peak_allocated = self.peak_allocated.max(self.total_allocated);

        let block = MemoryBlock {
            offset,
            size,
            compressed: false,
        };
        self.allocations.push(block.clone());

        Ok(MemoryHandle {
            gpu_id,
            pool_type: self.pool_type,
            block,
        })
    }

    /// Deallocate the block referenced by `handle`.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn deallocate(&mut self, handle: MemoryHandle) -> KwaversResult<()> {
        if let Some(index) = self
            .allocations
            .iter()
            .position(|b| b.offset == handle.block.offset)
        {
            let block = &self.allocations[index];
            self.total_allocated = self.total_allocated.saturating_sub(block.size);
            self.allocations.remove(index);
            Ok(())
        } else {
            Err("Memory block not found".into())
        }
    }

    /// Return per-pool allocation statistics.
    pub fn statistics(&self) -> GpuMemoryPoolStats {
        GpuMemoryPoolStats {
            allocated_bytes: self.total_allocated,
            peak_bytes: self.peak_allocated,
            transfer_count: 0,
            transfer_bytes: 0,
            compression_ratio: 1.0,
        }
    }
}
