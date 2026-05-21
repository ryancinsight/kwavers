//! GPU memory management: pools, buffers, and statistics.

use crate::core::error::{KwaversError, KwaversResult};
use std::collections::HashMap;

mod pool;

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
pub enum PinnGpuMemoryPoolType {
    Temporary,
    Persistent,
    Gradients,
    Collocation,
}

/// GPU memory manager with pool allocation
#[derive(Debug)]
pub struct GpuMemoryManager {
    pub(super) pools: HashMap<PinnGpuMemoryPoolType, PinnGpuAcceleratorMemoryPool>,
    pinned_buffers: Vec<PinnedBuffer<f32>>,
    transfer_streams: Vec<CudaStream>,
    stats: PinnGpuMemoryStats,
}

/// Memory pool for efficient allocation
#[derive(Debug)]
pub(crate) struct PinnGpuAcceleratorMemoryPool {
    pub(crate) pool_type: PinnGpuMemoryPoolType,
    pub(crate) total_allocated: usize,
    pub(crate) used_memory: usize,
    pub(crate) free_blocks: Vec<MemoryBlock>,
    pub(crate) alignment: usize,
}

/// Memory block in a pool
#[derive(Debug, Clone)]
pub(crate) struct MemoryBlock {
    pub(crate) ptr: *mut f32,
    pub(crate) size: usize,
    pub(crate) offset: usize,
}

/// Pinned host buffer for fast GPU transfers
#[derive(Debug, Clone)]
pub struct PinnedBuffer<T> {
    pub _ptr: *mut T,
    pub _size: usize,
}

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct PinnGpuMemoryStats {
    pub peak_gpu_memory: usize,
    pub current_gpu_memory: usize,
    pub peak_pinned_memory: usize,
    pub allocation_count: usize,
    pub deallocation_count: usize,
}

impl GpuMemoryManager {
    /// New.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new() -> KwaversResult<Self> {
        let mut pools = HashMap::new();

        pools.insert(
            PinnGpuMemoryPoolType::Temporary,
            PinnGpuAcceleratorMemoryPool::new(
                PinnGpuMemoryPoolType::Temporary,
                256 * 1024 * 1024,
                256,
            ),
        );
        pools.insert(
            PinnGpuMemoryPoolType::Persistent,
            PinnGpuAcceleratorMemoryPool::new(
                PinnGpuMemoryPoolType::Persistent,
                512 * 1024 * 1024,
                256,
            ),
        );
        pools.insert(
            PinnGpuMemoryPoolType::Gradients,
            PinnGpuAcceleratorMemoryPool::new(
                PinnGpuMemoryPoolType::Gradients,
                256 * 1024 * 1024,
                256,
            ),
        );
        pools.insert(
            PinnGpuMemoryPoolType::Collocation,
            PinnGpuAcceleratorMemoryPool::new(
                PinnGpuMemoryPoolType::Collocation,
                128 * 1024 * 1024,
                256,
            ),
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
            stats: PinnGpuMemoryStats::default(),
        })
    }
    /// Allocate device.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn allocate_device(
        &mut self,
        pool_type: PinnGpuMemoryPoolType,
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
    /// Deallocate device.
    /// # Errors
    /// - Returns [`KwaversError::System`] if the precondition for a System-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn deallocate_device(&mut self, buffer: CudaBuffer<f32>) -> KwaversResult<()> {
        let pool_type = match buffer.pool_id {
            0 => PinnGpuMemoryPoolType::Temporary,
            1 => PinnGpuMemoryPoolType::Persistent,
            2 => PinnGpuMemoryPoolType::Gradients,
            3 => PinnGpuMemoryPoolType::Collocation,
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
    /// Allocate pinned.
    /// # Errors
    /// - Returns [`KwaversError::System`] if the precondition for a System-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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
    /// Prefetch to device.
    /// # Errors
    /// - Returns [`KwaversError::System`] if the precondition for a System-class constraint is violated.
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
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
    /// Memory stats.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn memory_stats(&self) -> &PinnGpuMemoryStats {
        &self.stats
    }
    /// Defragment.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn defragment(&mut self) -> KwaversResult<()> {
        for pool in self.pools.values_mut() {
            pool.defragment()?;
        }
        Ok(())
    }
}
