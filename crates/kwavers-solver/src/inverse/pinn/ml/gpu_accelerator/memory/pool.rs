use kwavers_core::error::{KwaversError, KwaversResult};

use super::{MemoryBlock, PinnGpuAcceleratorMemoryPool, PinnGpuMemoryPoolType};

impl PinnGpuAcceleratorMemoryPool {
    /// New.
    pub(crate) fn new(
        pool_type: PinnGpuMemoryPoolType,
        total_size: usize,
        alignment: usize,
    ) -> Self {
        Self {
            pool_type,
            total_allocated: total_size,
            used_memory: 0,
            free_blocks: vec![MemoryBlock {
                ptr: std::ptr::null_mut(),
                size: total_size,
                offset: 0,
            }],
            alignment,
        }
    }
    /// Allocate.
    /// # Errors
    /// - Returns [`KwaversError::System`] if the precondition for a System-class constraint is violated.
    ///
    pub(crate) fn allocate(&mut self, size: usize) -> KwaversResult<MemoryBlock> {
        let aligned_size = size.div_ceil(self.alignment) * self.alignment;

        if aligned_size > self.total_allocated - self.used_memory {
            return Err(KwaversError::System(
                kwavers_core::error::SystemError::ResourceUnavailable {
                    resource: format!("memory pool {:?}", self.pool_type),
                },
            ));
        }

        for (i, block) in self.free_blocks.iter().enumerate() {
            if block.size >= aligned_size {
                let remaining = block.size - aligned_size;
                let allocated_block = MemoryBlock {
                    // SAFETY: block.ptr is valid and adding offset stays within the allocated block.
                    #[allow(unsafe_code)]
                    ptr: unsafe { block.ptr.add(block.offset) },
                    size: aligned_size,
                    offset: block.offset,
                };

                if remaining > 0 {
                    self.free_blocks[i] = MemoryBlock {
                        ptr: block.ptr,
                        size: remaining,
                        offset: block.offset + aligned_size,
                    };
                } else {
                    self.free_blocks.remove(i);
                }

                self.used_memory += aligned_size;
                return Ok(allocated_block);
            }
        }

        Err(KwaversError::System(
            kwavers_core::error::SystemError::ResourceUnavailable {
                resource: format!("memory pool {:?}", self.pool_type),
            },
        ))
    }
    /// Deallocate.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(crate) fn deallocate(&mut self, block: MemoryBlock) -> KwaversResult<()> {
        let block_size = block.size;
        self.free_blocks.push(block);
        self.used_memory -= block_size;
        Ok(())
    }
    /// Defragment.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(crate) fn defragment(&mut self) -> KwaversResult<()> {
        self.free_blocks.sort_by_key(|b| b.offset);

        let mut i = 0;
        while i < self.free_blocks.len().saturating_sub(1) {
            let (left, right) = self.free_blocks.split_at_mut(i + 1);
            let current = &left[i];
            let next = &right[0];

            if current.offset + current.size == next.offset {
                self.free_blocks[i] = MemoryBlock {
                    ptr: current.ptr,
                    size: current.size + next.size,
                    offset: current.offset,
                };
                self.free_blocks.remove(i + 1);
            } else {
                i += 1;
            }
        }

        Ok(())
    }
}
