//! Lock-Free Buffer Pool with NUMA Awareness
//!
//! - 64-byte cache-line alignment: optimal SIMD/cache performance
//! - NUMA awareness: delegates topology to [`super::numa`]
//! - Batch allocation: atomic multi-buffer acquisition
//! - Lock-free operations: Treiber stack for O(1) acquire/release
//!
//! # Algorithm
//!
//! **Theorem** (Treiber Stack Correctness): The CAS-based stack provides a
//! linearizable, lock-free LIFO queue under any interleaving of concurrent
//! operations.
//!
//! # References
//!
//! - Treiber R.K. (1986). "Systems programming: coping with parallelism".
//! - Bonwick J. (1994). "The Slab Allocator". *USENIX Summer Technical Conference*.

mod batch;
mod pool_impl;
#[cfg(test)]
mod tests;

pub use batch::{BufferBatch, NumaPoolManager};
pub use pool_impl::{BufferPool, PooledBuffer};

use crate::error::{KwaversError, KwaversResult};
use std::alloc::{alloc, Layout};
use std::ptr::NonNull;

/// Cache line size for x86_64 architectures (64 bytes).
pub const CACHE_LINE_SIZE: usize = 64;

/// Default pool capacity (number of buffers).
pub const DEFAULT_POOL_CAPACITY: usize = 16;

/// Buffer pool configuration.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Number of elements per buffer.
    pub elements: usize,
    /// Element size in bytes.
    pub element_size: usize,
    /// Number of buffers in pool.
    pub capacity: usize,
    /// NUMA node affinity (-1 for any node).
    pub numa_node: i32,
}

impl PoolConfig {
    /// Configuration for scalar field buffers with NUMA awareness.
    ///
    /// `T` determines the element size: `element_size = size_of::<T>()`.
    ///
    /// **Size Calculation**: `total_size = ceil(n × size_of::<T>() / 64) × 64`
    #[must_use]
    pub fn for_scalar_field<T: Sized>(elements: usize, capacity: usize, numa_node: i32) -> Self {
        Self {
            elements,
            element_size: std::mem::size_of::<T>(),
            capacity,
            numa_node,
        }
    }

    /// Total size per buffer including alignment padding.
    #[must_use]
    pub fn buffer_size(&self) -> usize {
        let raw_size = self.elements * self.element_size;
        (raw_size + CACHE_LINE_SIZE - 1) & !(CACHE_LINE_SIZE - 1)
    }

    /// Total memory required for the pool.
    #[must_use]
    pub fn total_memory_bytes(&self) -> usize {
        self.buffer_size() * self.capacity
    }
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            elements: 1024,
            element_size: std::mem::size_of::<f64>(),
            capacity: DEFAULT_POOL_CAPACITY,
            numa_node: -1,
        }
    }
}

/// Pool allocation statistics.
#[derive(Debug, Clone, Copy)]
pub struct PoolStats {
    /// Total capacity.
    pub capacity: usize,
    /// Currently allocated.
    pub allocated: usize,
    /// Peak allocation count.
    pub peak_allocated: usize,
    /// Buffer size in bytes.
    pub buffer_size: usize,
    /// Total memory reserved.
    pub total_memory: usize,
}

impl PoolStats {
    /// Current utilization ratio [0.0, 1.0].
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn utilization(&self) -> f64 {
        if self.capacity == 0 {
            0.0
        } else {
            self.allocated as f64 / self.capacity as f64
        }
    }
}

/// Allocate memory with 64-byte alignment.
/// # Errors
/// - Propagates any `KwaversError` returned by called functions.
///
pub(super) fn allocate_numa_aware(size: usize) -> KwaversResult<NonNull<u8>> {
    let layout = Layout::from_size_align(size, CACHE_LINE_SIZE).map_err(|_| {
        KwaversError::System(crate::error::SystemError::MemoryAllocation {
            requested_bytes: size,
            reason: "Invalid layout for NUMA-aware allocation".to_owned(),
        })
    })?;

    // SAFETY: Layout is valid (non-zero size, power-of-2 alignment).
    let ptr = unsafe { alloc(layout) };
    let memory = NonNull::new(ptr).ok_or_else(|| {
        KwaversError::System(crate::error::SystemError::MemoryAllocation {
            requested_bytes: size,
            reason: "Failed to allocate NUMA-aware memory".to_owned(),
        })
    })?;

    Ok(memory)
}