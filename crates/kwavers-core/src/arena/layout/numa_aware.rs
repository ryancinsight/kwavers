use mnemosyne_arena::{allocate_large_or_huge, deallocate_large_or_huge};
use mnemosyne_backend::MemoryBackendWrapper;
use mnemosyne_core::constants::SEGMENT_ALIGN;
use mnemosyne_core::types::Segment;
use moirai_parallel::{for_each_chunk_mut_with, Adaptive};
use std::ptr::NonNull;

#[cfg(test)]
use super::CACHE_LINE_SIZE;
use super::{ArenaLayoutNumaPolicy, NUMA_ALIGNMENT};
use crate::arena::numa::memory::{bind_memory_to_node, allocate_interleaved_memory};
use crate::error::{KwaversError, KwaversResult, SystemError};

// NUMA-AWARE MEMORY ALLOCATION
// ═══════════════════════════════════════════════════════════════════════════

/// Tracked allocation for NUMA-aware deallocation.
struct Allocation {
    user_ptr: NonNull<u8>,
    segment_ptr: *mut Segment,
}

impl Allocation {
    /// Create a new tracked allocation.
    fn new(user_ptr: NonNull<u8>, segment_ptr: *mut Segment) -> Self {
        Self {
            user_ptr,
            segment_ptr,
        }
    }

    /// Deallocate this tracked allocation.
    ///
    /// # Safety
    /// The stored pointers must remain valid and were originally returned
    /// by `allocate_large_or_huge`.
    unsafe fn deallocate(&self) {
        if !self.segment_ptr.is_null() {
            let _released = deallocate_large_or_huge::<MemoryBackendWrapper>(
                self.user_ptr.as_ptr(),
                self.segment_ptr,
            );
        }
    }
}

/// NUMA-aware memory allocation with support for multiple policies.
///
/// # Policies
///
/// - **FirstTouch**: Memory is not pre-bound to any NUMA node. On first write,
///   the OS allocates pages on the accessing thread's node. This is the default
///   and most portable strategy.
/// - **BindToNode(n)**: Memory is explicitly bound to NUMA node `n` using
///   `mbind(2)` on Linux. On non-Linux platforms, this behaves as FirstTouch.
/// - **Interleaved**: (Deferred) Memory would be interleaved across multiple
///   nodes; currently treated as FirstTouch.
///
/// # Platform Support
///
/// - **Linux**: `BindToNode` uses `mbind(2)` for explicit binding.
///   `Interleaved` requires allocator-level interleaver (planned).
/// - **Non-Linux**: All policies fall back to FirstTouch behavior.
#[derive(Debug)]
pub struct NumaAwareAllocator {
    /// Target NUMA node(s) for allocation
    policy: ArenaLayoutNumaPolicy,
    /// All tracked allocations for deallocation
    allocations: Vec<Allocation>,
}

impl NumaAwareAllocator {
    /// Create allocator with specified NUMA policy
    #[must_use]
    pub fn with_policy(policy: ArenaLayoutNumaPolicy) -> Self {
        Self {
            policy,
            allocations: Vec::new(),
        }
    }

    /// Allocate memory with NUMA awareness
    ///
    /// # Mathematical Specification
    ///
    /// **Precondition**: $\text{size} > 0 \land \text{align}$ is power of 2
    /// **Postcondition**: Returned pointer is $\text{align}$-byte aligned
    ///                    and satisfies the configured NUMA policy
    ///
    /// # Placement Guarantees
    ///
    /// - **FirstTouch**: Pages are allocated unbound; OS places them on first access
    /// - **BindToNode(n)**: Pages are bound to NUMA node `n` via `mbind(2)` (Linux only)
    /// - **Interleaved**: Currently treated as FirstTouch; deferred for future work
    ///
    /// # Errors
    /// - Returns [`KwaversError::System`] if `size` or `align` violate an
    ///   allocation-layout precondition.
    pub fn allocate(&mut self, size: usize, align: usize) -> KwaversResult<NonNull<u8>> {
        // Validate the caller's alignment request up front so an invalid `align`
        // surfaces with a descriptive error rather than the generic null-pointer
        // fallback from mnemosyne. mnemosyne always returns `SEGMENT_ALIGN`-
        // aligned memory (`SEGMENT_ALIGN >= NUMA_ALIGNMENT`), so the request is
        // satisfied whenever `align <= SEGMENT_ALIGN`.
        if !align.is_power_of_two() || align > SEGMENT_ALIGN {
            return Err(KwaversError::System(SystemError::MemoryAllocation {
                requested_bytes: size,
                reason: "Invalid align for NUMA allocation".to_owned(),
            }));
        }
        // `NUMA_ALIGNMENT` is the historical floor this allocator enforced; keep
        // the bound in the contract even though mnemosyne over-satisfies it.
        debug_assert!(align.max(NUMA_ALIGNMENT) <= SEGMENT_ALIGN);

        // SAFETY: `size <= MAX_ALLOC_SIZE` and `SEGMENT_ALIGN` alignment are
        // validated by mnemosyne's `is_valid_alloc_request`; a null user pointer
        // is mapped to the descriptive allocation error below.
        let user_ptr =
            unsafe { allocate_large_or_huge::<MemoryBackendWrapper>(size, SEGMENT_ALIGN, false) };
        let ptr = NonNull::new(user_ptr).ok_or_else(|| {
            KwaversError::System(SystemError::MemoryAllocation {
                requested_bytes: size,
                reason: "NUMA memory allocation failed".to_owned(),
            })
        })?;

        // SAFETY: Valid pointer to allocated memory.
        // Segment pointer is stored in metadata slot by allocate_large_or_huge.
        let segment_ptr = unsafe { *((ptr.as_ptr() as *mut *mut Segment).sub(1)) };

        // Apply NUMA policy to the allocated region.
        match self.policy {
            ArenaLayoutNumaPolicy::FirstTouch => {
                // No explicit binding; OS handles on first access.
            }
            ArenaLayoutNumaPolicy::BindToNode(node) => {
                // Explicitly bind to the requested node (Linux only; no-op elsewhere).
                // SAFETY: ptr is valid for size bytes.
                let bind_result = unsafe { bind_memory_to_node(ptr.as_ptr(), size, node) };
                if let Err(e) = bind_result {
                    // Log the error but don't fail allocation; fall back to FirstTouch.
                    eprintln!("Warning: Failed to bind memory to node {}: {:?}", node, e);
                }
            }
            ArenaLayoutNumaPolicy::Interleaved => {
                // Deferred: would require allocator-level interleaver.
                // Currently falls back to FirstTouch.
            }
        }

        // Track this allocation for safe deallocation.
        self.allocations.push(Allocation::new(ptr, segment_ptr));

        Ok(ptr)
    }

    /// Perform parallel first-touch initialization
    ///
    /// Divides allocated memory into chunks and has each thread in
    /// the thread pool initialize its chunk, establishing NUMA affinity.
    pub fn first_touch_parallel<T: Send + Copy + Default>(
        &self,
        ptr: NonNull<T>,
        num_elements: usize,
    ) {
        // Only perform if policy is FirstTouch
        if !matches!(self.policy, ArenaLayoutNumaPolicy::FirstTouch) {
            return;
        }

        // SAFETY: Memory is valid for num_elements * sizeof(T) bytes
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), num_elements) };

        let workers = std::thread::available_parallelism().map_or(1, usize::from);
        let chunk_size = num_elements.div_ceil(workers).max(1);

        for_each_chunk_mut_with::<Adaptive, _, _>(slice, chunk_size, |chunk| {
            chunk.fill(T::default());
        });
    }

    /// Get current NUMA policy
    #[inline]
    #[must_use]
    pub fn policy(&self) -> ArenaLayoutNumaPolicy {
        self.policy
    }
}

impl Drop for NumaAwareAllocator {
    fn drop(&mut self) {
        // Deallocate all tracked allocations in reverse order of allocation.
        while let Some(alloc) = self.allocations.pop() {
            // SAFETY: segment_ptr was written by allocate_large_or_huge and is valid.
            unsafe {
                alloc.deallocate();
            }
        }
    }
}

impl Default for NumaAwareAllocator {
    fn default() -> Self {
        Self {
            policy: ArenaLayoutNumaPolicy::FirstTouch,
            allocations: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numa_allocator() {
        let mut alloc = NumaAwareAllocator::with_policy(ArenaLayoutNumaPolicy::FirstTouch);

        let ptr = alloc
            .allocate(1024, CACHE_LINE_SIZE)
            .expect("allocation must succeed");
        // ptr is NonNull<u8>, so non-null is guaranteed by type invariant

        // First touch
        alloc.first_touch_parallel(ptr, 128usize);
    }
}
