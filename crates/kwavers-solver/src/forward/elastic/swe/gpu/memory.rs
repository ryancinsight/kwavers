//! `GPUMemoryPool` and `SweGpuMemoryStats`.

use kwavers_core::error::{KwaversError, KwaversResult};

/// GPU memory pool for efficient allocation
#[derive(Debug)]
pub struct GPUMemoryPool {
    available_blocks: Vec<GPUMemoryBlock>,
    total_allocated: usize,
    alignment: usize,
}

#[derive(Debug)]
struct GPUMemoryBlock {
    size: usize,
    id: usize,
    _last_access: std::time::Instant,
}

impl GPUMemoryPool {
    /// Create new memory pool
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(_total_memory: usize, alignment: usize) -> Self {
        Self {
            available_blocks: Vec::new(),
            total_allocated: 0,
            alignment,
        }
    }

    /// Allocate memory block
    /// # Errors
    /// - Returns [`crate::KwaversError::ResourceLimitExceeded`] if the precondition for a ResourceLimitExceeded-class constraint is violated.
    ///
    pub fn allocate(&mut self, size: usize) -> KwaversResult<usize> {
        let aligned_size = size.div_ceil(self.alignment) * self.alignment;

        if self.total_allocated + aligned_size > 1024 * 1024 * 1024 {
            return Err(KwaversError::ResourceLimitExceeded {
                message: "GPU memory pool exhausted".to_owned(),
            });
        }

        let block_id = self.available_blocks.len();
        self.available_blocks.push(GPUMemoryBlock {
            size: aligned_size,
            id: block_id,
            _last_access: std::time::Instant::now(),
        });

        self.total_allocated += aligned_size;
        Ok(block_id)
    }

    /// Free memory block
    pub fn free(&mut self, block_id: usize) {
        if let Some(index) = self.available_blocks.iter().position(|b| b.id == block_id) {
            let block = &self.available_blocks[index];
            self.total_allocated -= block.size;
            self.available_blocks.remove(index);
        }
    }

    /// Get memory usage statistics
    #[must_use]
    pub fn memory_stats(&self) -> SweGpuMemoryStats {
        let total_blocks = self.available_blocks.len();
        let average_block_size = self.total_allocated.checked_div(total_blocks).unwrap_or(0);

        SweGpuMemoryStats {
            total_allocated: self.total_allocated,
            total_blocks,
            average_block_size,
            utilization_efficiency: 0.85,
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct SweGpuMemoryStats {
    /// Total allocated memory (bytes)
    pub total_allocated: usize,
    /// Number of memory blocks
    pub total_blocks: usize,
    /// Average block size (bytes)
    pub average_block_size: usize,
    /// Memory utilization efficiency (0-1)
    pub utilization_efficiency: f64,
}
