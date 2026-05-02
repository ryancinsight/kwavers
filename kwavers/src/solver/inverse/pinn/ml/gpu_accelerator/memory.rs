//! GPU memory management: pools, buffers, and statistics.

use crate::core::error::{KwaversError, KwaversResult};
use std::collections::HashMap;

/// CUDA device buffer with memory management
#[derive(Debug)]
pub struct CudaBuffer<T> {
    pub ptr: *mut T,
    pub size: usize,
    pub pool_id: usize,
    pub stream: CudaStream,
}

/// CUDA stream for asynchronous operations
#[derive(Debug, Clone)]
pub struct CudaStream {
    pub handle: usize,
    pub priority: i32,
}

/// CUDA memory pool types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MemoryPoolType {
    Temporary,
    Persistent,
    Gradients,
    Collocation,
}

/// GPU memory manager with pool allocation
#[derive(Debug)]
pub struct GpuMemoryManager {
    pub(super) pools: HashMap<MemoryPoolType, MemoryPool>,
    pinned_buffers: Vec<PinnedBuffer<f32>>,
    transfer_streams: Vec<CudaStream>,
    stats: MemoryStats,
}

/// Memory pool for efficient allocation
#[derive(Debug)]
pub(super) struct MemoryPool {
    pub(super) pool_type: MemoryPoolType,
    pub(super) total_allocated: usize,
    pub(super) used_memory: usize,
    pub(super) free_blocks: Vec<MemoryBlock>,
    pub(super) alignment: usize,
}

/// Memory block in a pool
#[derive(Debug, Clone)]
pub(super) struct MemoryBlock {
    pub(super) ptr: *mut f32,
    pub(super) size: usize,
    pub(super) offset: usize,
}

/// Pinned host buffer for fast GPU transfers
#[derive(Debug, Clone)]
pub struct PinnedBuffer<T> {
    pub _ptr: *mut T,
    pub _size: usize,
}

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    pub peak_gpu_memory: usize,
    pub current_gpu_memory: usize,
    pub peak_pinned_memory: usize,
    pub allocation_count: usize,
    pub deallocation_count: usize,
}

impl GpuMemoryManager {
    pub fn new() -> KwaversResult<Self> {
        let mut pools = HashMap::new();

        pools.insert(
            MemoryPoolType::Temporary,
            MemoryPool::new(MemoryPoolType::Temporary, 256 * 1024 * 1024, 256),
        );
        pools.insert(
            MemoryPoolType::Persistent,
            MemoryPool::new(MemoryPoolType::Persistent, 512 * 1024 * 1024, 256),
        );
        pools.insert(
            MemoryPoolType::Gradients,
            MemoryPool::new(MemoryPoolType::Gradients, 256 * 1024 * 1024, 256),
        );
        pools.insert(
            MemoryPoolType::Collocation,
            MemoryPool::new(MemoryPoolType::Collocation, 128 * 1024 * 1024, 256),
        );

        let transfer_streams = (0..4)
            .map(|i| CudaStream {
                handle: i,
                priority: 0,
            })
            .collect();

        Ok(Self {
            pools,
            pinned_buffers: Vec::new(),
            transfer_streams,
            stats: MemoryStats::default(),
        })
    }

    pub fn allocate_device(
        &mut self,
        pool_type: MemoryPoolType,
        size: usize,
    ) -> KwaversResult<CudaBuffer<f32>> {
        let pool = self.pools.get_mut(&pool_type).ok_or_else(|| {
            KwaversError::System(crate::core::error::SystemError::ResourceUnavailable {
                resource: format!("memory pool {:?}", pool_type),
            })
        })?;

        let block = pool.allocate(size * std::mem::size_of::<f32>())?;

        self.stats.allocation_count += 1;
        self.stats.current_gpu_memory += block.size;
        self.stats.peak_gpu_memory = self
            .stats
            .peak_gpu_memory
            .max(self.stats.current_gpu_memory);

        Ok(CudaBuffer {
            ptr: block.ptr,
            size,
            pool_id: pool_type as usize,
            stream: CudaStream {
                handle: 0,
                priority: 0,
            },
        })
    }

    pub fn deallocate_device(&mut self, buffer: CudaBuffer<f32>) -> KwaversResult<()> {
        let pool_type = match buffer.pool_id {
            0 => MemoryPoolType::Temporary,
            1 => MemoryPoolType::Persistent,
            2 => MemoryPoolType::Gradients,
            3 => MemoryPoolType::Collocation,
            _ => {
                return Err(KwaversError::System(
                    crate::core::error::SystemError::InvalidConfiguration {
                        parameter: "pool_id".to_string(),
                        reason: "Invalid memory pool ID".to_string(),
                    },
                ))
            }
        };

        let pool = self.pools.get_mut(&pool_type).unwrap();
        pool.deallocate(MemoryBlock {
            ptr: buffer.ptr,
            size: buffer.size * std::mem::size_of::<f32>(),
            offset: 0,
        })?;

        self.stats.deallocation_count += 1;
        self.stats.current_gpu_memory -= buffer.size * std::mem::size_of::<f32>();

        Ok(())
    }

    pub fn allocate_pinned(&mut self, size: usize) -> KwaversResult<PinnedBuffer<f32>> {
        let layout = std::alloc::Layout::array::<f32>(size).map_err(|_| {
            KwaversError::System(crate::core::error::SystemError::ResourceUnavailable {
                resource: "pinned memory allocation".to_string(),
            })
        })?;

        // SAFETY: layout is valid and the allocation is checked for null immediately after.
        #[allow(unsafe_code)]
        let ptr = unsafe { std::alloc::alloc(layout) as *mut f32 };

        if ptr.is_null() {
            return Err(KwaversError::System(
                crate::core::error::SystemError::ResourceUnavailable {
                    resource: "pinned memory".to_string(),
                },
            ));
        }

        let buffer = PinnedBuffer {
            _ptr: ptr,
            _size: size,
        };

        self.stats.peak_pinned_memory = self
            .stats
            .peak_pinned_memory
            .max(size * std::mem::size_of::<f32>());
        self.pinned_buffers.push(buffer.clone());

        Ok(buffer)
    }

    pub fn prefetch_to_device(
        &self,
        host_data: &[f32],
        device_buffer: &CudaBuffer<f32>,
        _stream_idx: usize,
    ) -> KwaversResult<()> {
        if _stream_idx >= self.transfer_streams.len() {
            return Err(KwaversError::System(
                crate::core::error::SystemError::InvalidConfiguration {
                    parameter: "_stream_idx".to_string(),
                    reason: "Invalid transfer stream index".to_string(),
                },
            ));
        }

        if host_data.len() != device_buffer.size {
            return Err(KwaversError::Validation(
                crate::core::error::ValidationError::FieldValidation {
                    field: "data_size".to_string(),
                    value: host_data.len().to_string(),
                    constraint: format!("must match device buffer size {}", device_buffer.size),
                },
            ));
        }

        // SAFETY: host_data and device_buffer pointers are valid and have the same length.
        #[allow(unsafe_code)]
        unsafe {
            std::ptr::copy_nonoverlapping(host_data.as_ptr(), device_buffer.ptr, host_data.len());
        }

        Ok(())
    }

    pub fn memory_stats(&self) -> &MemoryStats {
        &self.stats
    }

    pub fn defragment(&mut self) -> KwaversResult<()> {
        for pool in self.pools.values_mut() {
            pool.defragment()?;
        }
        Ok(())
    }
}

impl MemoryPool {
    pub(super) fn new(pool_type: MemoryPoolType, total_size: usize, alignment: usize) -> Self {
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

    pub(super) fn allocate(&mut self, size: usize) -> KwaversResult<MemoryBlock> {
        let aligned_size = size.div_ceil(self.alignment) * self.alignment;

        if aligned_size > self.total_allocated - self.used_memory {
            return Err(KwaversError::System(
                crate::core::error::SystemError::ResourceUnavailable {
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
            crate::core::error::SystemError::ResourceUnavailable {
                resource: format!("memory pool {:?}", self.pool_type),
            },
        ))
    }

    pub(super) fn deallocate(&mut self, block: MemoryBlock) -> KwaversResult<()> {
        let block_size = block.size;
        self.free_blocks.push(block);
        self.used_memory -= block_size;
        Ok(())
    }

    pub(super) fn defragment(&mut self) -> KwaversResult<()> {
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
