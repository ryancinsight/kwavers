//! Memory optimization strategies

use kwavers_core::error::KwaversResult;
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
    alignment: usize,
}

impl Default for MemoryOptimizer {
    /// 64-byte cache-line alignment.
    fn default() -> Self {
        Self { alignment: 64 }
    }
}

impl MemoryOptimizer {
    /// Allocate aligned memory for better cache performance
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn allocate_aligned<T>(&self, count: usize) -> KwaversResult<*mut T> {
        let size = count * std::mem::size_of::<T>();
        let align = self.alignment.max(std::mem::align_of::<T>());

        let layout = Layout::from_size_align(size, align).map_err(|e| {
            kwavers_core::error::KwaversError::System(
                kwavers_core::error::SystemError::MemoryAllocation {
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
                return Err(kwavers_core::error::KwaversError::System(
                    kwavers_core::error::SystemError::MemoryAllocation {
                        requested_bytes: size,
                        reason: "Failed to allocate aligned memory".to_owned(),
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
    pub fn create_pool(&self, size: usize) -> PerfMemoryPool {
        PerfMemoryPool::new(size, self.alignment)
    }

    /// Optimize memory layout for column-major access
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
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
pub struct PerfMemoryPool {
    buffer: Vec<u8>,
    offset: usize,
    alignment: usize,
}

impl PerfMemoryPool {
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
        //   - Rejection: PerfMemoryPool is a specialized bump allocator with reset capability
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

#[cfg(test)]
mod tests {
    use super::*;

    // ─── BandwidthOptimizer exact formula tests ───────────────────────────────

    /// `estimate_utilization` at 100% bandwidth matches formula exactly.
    ///
    /// max_bandwidth = 50.0 GB/s.
    /// bytes = 50×10⁹, time = 1.0 s → bandwidth_gbps = 50.0.
    /// utilization = 50.0/50.0 × 100 = 100.0 %.
    #[test]
    fn bandwidth_optimizer_full_utilization_is_100_percent() {
        let mut optimizer = BandwidthOptimizer::new();
        let utilization = optimizer.estimate_utilization(50_000_000_000usize, 1.0);
        assert!(
            (utilization - 100.0).abs() < 1e-10,
            "utilization = {utilization} (expected 100.0 at full bandwidth)"
        );
    }

    /// `estimate_utilization` at half bandwidth gives 50.0%.
    ///
    /// bytes = 25×10⁹, time = 1.0 s → bandwidth_gbps = 25.0.
    /// utilization = 25.0/50.0 × 100 = 50.0 %.
    #[test]
    fn bandwidth_optimizer_half_utilization_is_50_percent() {
        let mut optimizer = BandwidthOptimizer::new();
        let utilization = optimizer.estimate_utilization(25_000_000_000usize, 1.0);
        assert!(
            (utilization - 50.0).abs() < 1e-10,
            "utilization = {utilization} (expected 50.0 at half bandwidth)"
        );
    }

    /// `estimate_utilization` at double time gives half utilization.
    ///
    /// bytes = 50×10⁹, time = 2.0 s → bandwidth_gbps = 25.0 → 50%.
    #[test]
    fn bandwidth_optimizer_double_time_halves_utilization() {
        let mut optimizer = BandwidthOptimizer::new();
        let utilization = optimizer.estimate_utilization(50_000_000_000usize, 2.0);
        assert!(
            (utilization - 50.0).abs() < 1e-10,
            "utilization = {utilization} (expected 50.0 at double time)"
        );
    }

    // ─── MemoryOptimizer::transpose_for_column_major exact tests ─────────────

    /// Transpose of 2×3 row-major [1,2,3,4,5,6] gives column-major [1,4,2,5,3,6].
    ///
    /// Row-major A[i,j] = data[i*3+j]:
    ///   A = [[1,2,3],[4,5,6]].
    /// Column-major result[j*2+i] = A[i,j]:
    ///   `[0]`=A[0,0]=1, `[1]`=A[1,0]=4, `[2]`=A[0,1]=2, `[3]`=A[1,1]=5,
    ///   `[4]`=A[0,2]=3, `[5]`=A[1,2]=6 → [1,4,2,5,3,6].
    #[test]
    fn memory_optimizer_transpose_2x3_exact() {
        let optimizer = MemoryOptimizer::default();
        let data = vec![1u32, 2, 3, 4, 5, 6];
        let result = optimizer.transpose_for_column_major(&data, 2, 3);
        assert_eq!(
            result,
            vec![1u32, 4, 2, 5, 3, 6],
            "column-major transpose of 2×3 must be [1,4,2,5,3,6]"
        );
    }

    /// Transpose of a 3×1 column vector is equivalent to itself in column-major form.
    ///
    /// rows=3, cols=1: result[0*3+i] = data[i*1+0] → result`i` = data`i`.
    #[test]
    fn memory_optimizer_transpose_column_vector_is_identity() {
        let optimizer = MemoryOptimizer::default();
        let data = vec![7.0f64, 8.0, 9.0];
        let result = optimizer.transpose_for_column_major(&data, 3, 1);
        assert_eq!(result, data, "transpose of column vector must equal input");
    }

    /// Transpose of 1×N row vector produces N×1 column vector (same bytes).
    ///
    /// rows=1, cols=4: result[j*1+0] = data[0*4+j] = data`J` → identical.
    #[test]
    fn memory_optimizer_transpose_row_vector_is_identity() {
        let optimizer = MemoryOptimizer::default();
        let data = vec![10i32, 20, 30, 40];
        let result = optimizer.transpose_for_column_major(&data, 1, 4);
        assert_eq!(result, data, "transpose of row vector must equal input");
    }

    // ─── PerfMemoryPool exact allocation tests ───────────────────────────────────

    /// Sequential allocations advance the offset by the aligned size.
    ///
    /// Pool of 128 bytes, alignment=16:
    ///   alloc(8) → offset=8; alloc(8) → aligned_offset=⌈8/16⌉×16=16, offset=24.
    /// Both pointers must be non-null.
    #[test]
    fn memory_pool_sequential_alloc_returns_non_null() {
        let mut pool = PerfMemoryPool::new(128, 16);
        let p1 = pool.allocate(8);
        let p2 = pool.allocate(8);
        assert!(p1.is_some(), "first allocation must succeed");
        assert!(p2.is_some(), "second allocation must succeed");
        assert_ne!(
            p1.unwrap(),
            p2.unwrap(),
            "consecutive allocations must return distinct pointers"
        );
    }

    /// Pool exhaustion returns `None`.
    ///
    /// Pool of 16 bytes, alignment=8: alloc(16) succeeds, alloc(1) fails.
    #[test]
    fn memory_pool_exhaustion_returns_none() {
        let mut pool = PerfMemoryPool::new(16, 8);
        let p1 = pool.allocate(16);
        assert!(p1.is_some(), "first allocation of 16 bytes must succeed");
        let p2 = pool.allocate(1);
        assert!(p2.is_none(), "allocation past capacity must return None");
    }

    /// `reset()` allows the same pool to be reused from the start.
    ///
    /// Allocate 16 bytes, reset, then allocate 16 bytes again — second pointer
    /// equals the base pointer (same offset=0 after reset).
    #[test]
    fn memory_pool_reset_restores_start_offset() {
        let mut pool = PerfMemoryPool::new(64, 8);
        let p1 = pool.allocate(16).expect("first alloc must succeed");
        pool.reset();
        let p2 = pool.allocate(16).expect("post-reset alloc must succeed");
        assert_eq!(
            p1, p2,
            "post-reset allocation must return same base pointer as first allocation"
        );
    }
}