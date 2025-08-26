//! GPU memory allocation strategies

use super::buffer::GpuBuffer;
use crate::error::{KwaversError, KwaversResult};
use crate::gpu::GpuBackend;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// GPU memory allocation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationStrategy {
    /// On-demand allocation
    OnDemand,
    /// Pool-based allocation
    Pool,
    /// Streaming allocation
    Streaming,
    /// Unified memory allocation
    Unified,
}

/// GPU memory allocator
#[derive(Debug)]
pub struct Allocator {
    strategy: AllocationStrategy,
    allocated: AtomicUsize,
    peak: AtomicUsize,
    next_id: AtomicUsize,
}

impl Allocator {
    /// Create new allocator
    pub fn new(strategy: AllocationStrategy) -> Self {
        Self {
            strategy,
            allocated: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            next_id: AtomicUsize::new(0),
        }
    }

    /// Allocate GPU memory
    pub fn allocate(
        &self,
        size_bytes: usize,
        backend: &dyn GpuBackend,
    ) -> KwaversResult<GpuBuffer> {
        let device_ptr = backend.allocate_memory(size_bytes)?;

        let buffer = GpuBuffer {
            id: self.next_id.fetch_add(1, Ordering::SeqCst),
            size_bytes,
            device_ptr: Some(device_ptr as u64),
            host_ptr: None,
            is_pinned: false,
            allocation_time: Instant::now(),
            last_access_time: Instant::now(),
        };

        // Update statistics
        let allocated = self.allocated.fetch_add(size_bytes, Ordering::SeqCst) + size_bytes;
        let mut peak = self.peak.load(Ordering::SeqCst);
        while allocated > peak {
            match self.peak.compare_exchange_weak(
                peak,
                allocated,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => break,
                Err(x) => peak = x,
            }
        }

        Ok(buffer)
    }

    /// Free GPU memory
    pub fn free(&self, buffer: GpuBuffer, backend: &dyn GpuBackend) -> KwaversResult<()> {
        if let Some(device_ptr) = buffer.device_ptr {
            backend.free_memory(device_ptr as *mut u8)?;
            self.allocated
                .fetch_sub(buffer.size_bytes, Ordering::SeqCst);
        }
        Ok(())
    }

    /// Get allocated bytes
    pub fn allocated_bytes(&self) -> usize {
        self.allocated.load(Ordering::SeqCst)
    }

    /// Get peak allocated bytes
    pub fn peak_bytes(&self) -> usize {
        self.peak.load(Ordering::SeqCst)
    }
}
