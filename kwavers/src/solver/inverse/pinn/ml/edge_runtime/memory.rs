use super::{MemoryAllocator, MemoryBlock};
use crate::core::error::{KwaversError, KwaversResult};

impl MemoryAllocator {
    /// Create a new memory allocator
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(total_memory: usize) -> Self {
        Self {
            total_memory,
            allocations: Vec::new(),
            fragmentation_ratio: 0.0,
        }
    }

    /// Allocate a memory block with alignment
    /// # Errors
    /// - Returns [`KwaversError::System`] if the precondition for a System-class constraint is violated.
    ///
    pub fn allocate_block(&mut self, size: usize, alignment: usize) -> KwaversResult<usize> {
        let aligned_size = size.div_ceil(alignment) * alignment;

        let used_memory: usize = self.allocations.iter().map(|b| b.size).sum();
        if used_memory + aligned_size > self.total_memory {
            return Err(KwaversError::System(
                crate::core::error::SystemError::ResourceUnavailable {
                    resource: format!(
                        "Memory limit reached: {}/{} bytes",
                        used_memory + aligned_size,
                        self.total_memory
                    ),
                },
            ));
        }

        let mut best_start = 0;
        let mut allocations = self.allocations.clone();
        allocations.sort_by_key(|b| b.start_address);

        for block in &allocations {
            if block.start_address >= best_start + aligned_size {
                break;
            }
            best_start = (block.start_address + block.size).div_ceil(alignment) * alignment;
        }

        if best_start + aligned_size > self.total_memory {
            return Err(KwaversError::System(
                crate::core::error::SystemError::ResourceUnavailable {
                    resource: "Memory fragmentation: No contiguous block found".to_string(),
                },
            ));
        }

        let new_block = MemoryBlock {
            start_address: best_start,
            size: aligned_size,
            allocated: true,
            _alignment: alignment,
        };

        self.allocations.push(new_block);
        self.update_fragmentation_stats();

        Ok(best_start)
    }

    fn update_fragmentation_stats(&mut self) {
        if self.allocations.is_empty() {
            self.fragmentation_ratio = 0.0;
            return;
        }

        let total_allocated: usize = self.allocations.iter().map(|b| b.size).sum();
        let max_address = self
            .allocations
            .iter()
            .map(|b| b.start_address + b.size)
            .max()
            .unwrap_or(0);

        if max_address == 0 {
            self.fragmentation_ratio = 0.0;
        } else {
            self.fragmentation_ratio = 1.0 - (total_allocated as f32 / max_address as f32);
        }
    }

    /// Get total allocated memory
    pub fn get_allocated_memory(&self) -> usize {
        self.allocations
            .iter()
            .filter(|block| block.allocated)
            .map(|block| block.size)
            .sum()
    }
}
