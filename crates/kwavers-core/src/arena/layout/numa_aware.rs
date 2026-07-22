use moirai_parallel::{for_each_chunk_mut_with, Adaptive};
use std::alloc::{alloc, Layout};
use std::ptr::NonNull;

#[cfg(test)]
use super::CACHE_LINE_SIZE;
use super::{ArenaLayoutNumaPolicy, NUMA_ALIGNMENT};
use crate::error::{KwaversError, KwaversResult, SystemError};

// NUMA-AWARE MEMORY ALLOCATION
// ═══════════════════════════════════════════════════════════════════════════

/// NUMA-aware memory allocation policy
///
/// Implements first-touch allocation strategy where memory is allocated
/// on the NUMA node of the first thread to write to it.
#[derive(Debug)]
pub struct NumaAwareAllocator {
    /// Target NUMA node(s) for allocation
    policy: ArenaLayoutNumaPolicy,
}

impl NumaAwareAllocator {
    /// Create allocator with specified NUMA policy
    #[must_use]
    pub fn with_policy(policy: ArenaLayoutNumaPolicy) -> Self {
        Self { policy }
    }

    /// Allocate memory with NUMA awareness
    ///
    /// # Mathematical Specification
    ///
    /// **Precondition**: $\text{size} > 0 \land \text{align}$ is power of 2
    /// **Postcondition**: Returned pointer is $\text{align}$-byte aligned
    ///                    and suitable for first-touch NUMA optimization
    ///
    /// # Implementation Notes
    ///
    /// First-touch policy: Memory is not bound to any NUMA node initially.
    /// On first write, OS allocates pages on the accessing thread's node.
    /// This is achieved by:
    /// 1. Allocating with standard allocator (pages unbound)
    /// 2. Optionally touching pages in parallel across desired nodes
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn allocate(&self, size: usize, align: usize) -> KwaversResult<NonNull<u8>> {
        let layout = Layout::from_size_align(size, align.max(NUMA_ALIGNMENT)).map_err(|_| {
            KwaversError::System(SystemError::MemoryAllocation {
                requested_bytes: size,
                reason: "Invalid layout for NUMA allocation".to_owned(),
            })
        })?;

        // SAFETY: Layout is valid (checked above)
        let ptr = unsafe { alloc(layout) };

        NonNull::new(ptr).ok_or_else(|| {
            KwaversError::System(SystemError::MemoryAllocation {
                requested_bytes: size,
                reason: "NUMA memory allocation failed".to_owned(),
            })
        })
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

impl Default for NumaAwareAllocator {
    fn default() -> Self {
        Self {
            policy: ArenaLayoutNumaPolicy::FirstTouch,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numa_allocator() {
        let alloc = NumaAwareAllocator::with_policy(ArenaLayoutNumaPolicy::FirstTouch);

        let ptr = alloc
            .allocate(1024, CACHE_LINE_SIZE)
            .expect("allocation must succeed");
        // ptr is NonNull<u8>, so non-null is guaranteed by type invariant

        // First touch
        alloc.first_touch_parallel(ptr, 128usize);
    }
}