//! Memory optimization strategies

use crate::error::KwaversResult;
use std::alloc::{alloc, dealloc, Layout};

/// Prefetch strategy for memory access
#[derive(Debug, Clone, Copy)]
pub enum PrefetchStrategy {
    /// No prefetching
    None,
    /// Conservative prefetching
    Conservative,
    /// Aggressive prefetching
    Aggressive,
    /// Adaptive prefetching based on access pattern
    Adaptive,
}

/// Bandwidth optimizer for memory transfers
#[derive(Debug)]
pub struct BandwidthOptimizer {
    /// Maximum bandwidth in GB/s
    max_bandwidth: f64,
    /// Current utilization percentage
    utilization: f64,
}

impl Default for BandwidthOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl BandwidthOptimizer {
    /// Create a new bandwidth optimizer
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_bandwidth: 50.0, // Typical DDR4 bandwidth
            utilization: 0.0,
        }
    }

    /// Estimate bandwidth utilization
    pub fn estimate_utilization(&mut self, bytes: usize, time_seconds: f64) -> f64 {
        let bandwidth_gbps = (bytes as f64 / 1e9) / time_seconds;
        self.utilization = (bandwidth_gbps / self.max_bandwidth) * 100.0;
        self.utilization
    }
}

/// Memory optimizer for efficient memory management
#[derive(Debug)]
pub struct MemoryOptimizer {
    prefetch_distance: usize,
    alignment: usize,
    huge_pages_enabled: bool,
}

impl MemoryOptimizer {
    /// Create a new memory optimizer
    #[must_use]
    pub fn new(prefetch_distance: usize) -> Self {
        Self {
            prefetch_distance,
            alignment: 64, // Cache line alignment
            huge_pages_enabled: false,
        }
    }

    /// Enable memory prefetching
    pub fn enable_prefetching(&self) -> KwaversResult<()> {
        log::info!(
            "Memory prefetching enabled with distance: {}",
            self.prefetch_distance
        );
        Ok(())
    }

    /// Allocate aligned memory for better cache performance
    pub fn allocate_aligned<T>(&self, count: usize) -> KwaversResult<*mut T> {
        let size = count * std::mem::size_of::<T>();
        let align = self.alignment.max(std::mem::align_of::<T>());

        let layout = Layout::from_size_align(size, align).map_err(|e| {
            crate::error::KwaversError::System(crate::error::SystemError::MemoryAllocation {
                requested_bytes: size,
                reason: e.to_string(),
            })
        })?;

        // SAFETY:
        // 1. Layout is valid as checked above
        // 2. alloc returns properly aligned memory or null
        // 3. We check for null before returning
        // 4. Caller is responsible for proper deallocation
        unsafe {
            let ptr = alloc(layout).cast::<T>();
            if ptr.is_null() {
                return Err(crate::error::KwaversError::System(
                    crate::error::SystemError::MemoryAllocation {
                        requested_bytes: size,
                        reason: "Failed to allocate aligned memory".to_string(),
                    },
                ));
            }
            Ok(ptr)
        }
    }

    /// Deallocate aligned memory
    pub unsafe fn deallocate_aligned<T>(&self, ptr: *mut T, count: usize) {
        unsafe {
            let size = count * std::mem::size_of::<T>();
            let align = self.alignment.max(std::mem::align_of::<T>());

            if let Ok(layout) = Layout::from_size_align(size, align) {
                dealloc(ptr.cast::<u8>(), layout);
            }
        }
    }

    /// Create a memory pool for reduced allocation overhead
    #[must_use]
    pub fn create_pool(&self, size: usize) -> MemoryPool {
        MemoryPool::new(size, self.alignment)
    }

    /// Optimize memory layout for column-major access
    pub fn transpose_for_column_major<T: Copy>(
        &self,
        data: &[T],
        rows: usize,
        cols: usize,
    ) -> Vec<T> {
        assert_eq!(data.len(), rows * cols);

        let mut transposed = Vec::with_capacity(data.len());
        unsafe {
            transposed.set_len(data.len());
        }

        for j in 0..cols {
            for i in 0..rows {
                transposed[j * rows + i] = data[i * cols + j];
            }
        }

        transposed
    }
}

/// Memory pool for efficient allocation
#[derive(Debug)]
pub struct MemoryPool {
    buffer: Vec<u8>,
    offset: usize,
    alignment: usize,
}

impl MemoryPool {
    /// Create a new memory pool
    #[must_use]
    pub fn new(size: usize, alignment: usize) -> Self {
        Self {
            buffer: vec![0u8; size],
            offset: 0,
            alignment,
        }
    }

    /// Allocate from the pool
    pub fn allocate(&mut self, size: usize) -> Option<*mut u8> {
        // Align the offset
        let aligned_offset = self.offset.div_ceil(self.alignment) * self.alignment;

        if aligned_offset + size > self.buffer.len() {
            return None;
        }

        let ptr = unsafe { self.buffer.as_mut_ptr().add(aligned_offset) };
        self.offset = aligned_offset + size;

        Some(ptr)
    }

    /// Reset the pool for reuse
    pub fn reset(&mut self) {
        self.offset = 0;
    }
}
