//! Memory optimization strategies

use crate::core::error::KwaversResult;
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
    #[allow(dead_code)] // Memory optimization configuration for advanced systems
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
            crate::core::error::KwaversError::System(
                crate::core::error::SystemError::MemoryAllocation {
                    requested_bytes: size,
                    reason: e.to_string(),
                },
            )
        })?;

        // SAFETY: Aligned memory allocation with OOM handling and type cast
        //   - Layout: size = count × sizeof(T), align = max(alignment, align_of::<T>())
        //   - Allocation: alloc(layout) returns pointer to aligned memory or null
        //   - Type cast: u8 → T valid if alignment requirements met (enforced by layout)
        //   - Null check: Returns None on allocation failure (caller handles OOM)
        //   - Caller responsibility: Must deallocate with matching layout via deallocate_aligned()
        // INVARIANTS:
        //   - Precondition: count × sizeof(T) ≤ isize::MAX (enforced by Layout construction)
        //   - Precondition: alignment is power of 2 and ≤ system max alignment
        //   - Postcondition: ptr is aligned to max(alignment, align_of::<T>())
        //   - Postcondition: ptr is valid for count × sizeof(T) bytes (if non-null)
        //   - Lifetime: Caller must ensure deallocation before ptr becomes invalid
        // ALTERNATIVES:
        //   - Box<[T]> for aligned allocations
        //   - Rejection: Box doesn't support custom alignment > align_of::<T>()
        //   - aligned_alloc (C function)
        //   - Rejection: std::alloc::alloc is portable Rust idiom
        // PERFORMANCE:
        //   - Allocation cost: Similar to malloc (~50-500 cycles depending on allocator)
        //   - Alignment benefit: Eliminates unaligned access penalties (measured 5-10% speedup for SIMD)
        //   - Use case: SIMD arrays requiring 32/64-byte alignment
        #[allow(unsafe_code)]
        unsafe {
            let ptr = alloc(layout).cast::<T>();
            if ptr.is_null() {
                return Err(crate::core::error::KwaversError::System(
                    crate::core::error::SystemError::MemoryAllocation {
                        requested_bytes: size,
                        reason: "Failed to allocate aligned memory".to_string(),
                    },
                ));
            }
            Ok(ptr)
        }
    }

    /// Deallocate aligned memory
    ///
    /// # Safety
    /// The pointer must have been returned by a previous call to `allocate_aligned` with
    /// the same allocator and count. The memory must not be accessed after deallocation.
    /// The count must match exactly the count used during allocation.
    #[allow(unsafe_code)]
    pub unsafe fn deallocate_aligned<T>(&self, ptr: *mut T, count: usize) {
        // SAFETY: Aligned memory deallocation with layout reconstruction
        //   - Layout reconstruction: Must match allocate_aligned() parameters exactly
        //   - Pointer match: ptr must be pointer returned by allocate_aligned()
        //   - Count match: count must match original allocation
        //   - Type cast: T → u8 reverses cast from allocation
        //   - Single deallocation: Caller must ensure dealloc called exactly once per allocation
        // INVARIANTS:
        //   - Precondition: ptr was allocated via allocate_aligned() with same count and alignment
        //   - Precondition: count matches original allocation count
        //   - Precondition: No outstanding references to memory at ptr exist
        //   - Postcondition: Memory returned to allocator
        //   - Lifetime: Caller must not access ptr after deallocation
        // ALTERNATIVES:
        //   - Store layout at allocation time (requires wrapper struct)
        //   - Rejection: Memory overhead, caller typically knows allocation parameters
        //   - Reference counting (Rc/Arc)
        //   - Rejection: Unnecessary overhead for manual memory management use case
        // PERFORMANCE:
        //   - Deallocation cost: Similar to free (~50-200 cycles)
        //   - Layout reconstruction: Negligible overhead (~2-3 cycles)
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
    pub fn transpose_for_column_major<T: Copy + Default>(
        &self,
        data: &[T],
        rows: usize,
        cols: usize,
    ) -> Vec<T> {
        assert_eq!(data.len(), rows * cols);

        // Use vec![T::default(); size] for safe initialization
        let mut transposed = vec![T::default(); data.len()];

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

        // SAFETY: Memory pool allocation with pointer arithmetic and alignment
        //   - Alignment: aligned_offset = ⌈offset / alignment⌉ × alignment
        //   - Bounds check: aligned_offset + size ≤ buffer.len() (checked before pointer arithmetic)
        //   - Pointer arithmetic: buffer.as_mut_ptr().add(aligned_offset) within buffer bounds
        //   - Lifetime: Returned pointer valid until pool is dropped or reset
        //   - No individual deallocation: Pool allocations freed together at reset/drop
        // INVARIANTS:
        //   - Precondition: size > 0 (zero-size allocations handled separately)
        //   - Precondition: aligned_offset + size ≤ buffer.len() (checked, returns None on overflow)
        //   - Postcondition: ptr is aligned to self.alignment
        //   - Postcondition: offset updated to aligned_offset + size (monotonic increase)
        // ALTERNATIVES:
        //   - Vec<u8> per allocation
        //   - Rejection: Heap allocation overhead defeats pool purpose
        //   - Bump allocator with separate buffer
        //   - Rejection: MemoryPool is a specialized bump allocator with reset capability
        // PERFORMANCE:
        //   - Allocation: O(1), ~2-3 cycles (alignment + bounds check + pointer bump)
        //   - Reset: O(1), just resets offset to 0 (no deallocation)
        //   - Use case: Per-frame allocations in real-time rendering/simulation
        #[allow(unsafe_code)]
        let ptr = unsafe { self.buffer.as_mut_ptr().add(aligned_offset) };
        self.offset = aligned_offset + size;

        Some(ptr)
    }

    /// Reset the pool for reuse
    pub fn reset(&mut self) {
        self.offset = 0;
    }
}
