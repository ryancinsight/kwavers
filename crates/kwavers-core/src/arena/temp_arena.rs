// Temporary arenas — bump allocator and scoped slab arena.
//
// * [`BumpAllocator`]: O(1) sequential allocator; reset to reuse entire pool.
// * [`ScopedArena`]: `FieldArena` wrapper that tracks live slots for ordered
//   cleanup (useful when RAII field references cannot be returned to callers).
//
// # References
// - Evans J. (2006). BSDCan Conference, 157–168 (jemalloc arena design).

use mnemosyne_arena::{allocate_large_or_huge, deallocate_large_or_huge};
use mnemosyne_backend::MemoryBackendWrapper;
use mnemosyne_core::constants::SEGMENT_ALIGN;
use mnemosyne_core::types::Segment;
use std::cell::RefCell;
use std::ptr::NonNull;

use crate::error::{KwaversError, KwaversResult};

use super::field_arena::{ArenaConfig, ArenaStats, FieldArena};

// ─── BumpAllocator ────────────────────────────────────────────────────────────

/// High-performance bump (linear) allocator for sequential allocations.
///
/// Allocation is O(1): an integer offset is bumped forward.
/// Individual deallocation is unsupported; call [`reset`][Self::reset] to
/// reclaim the entire pool at once.
///
/// # Algorithm
///
/// **Theorem** (Alignment): For any alignment `a = 2^k` and offset `off`,
/// the aligned offset `aligned = (off + a - 1) & !(a - 1)` satisfies
/// `aligned ≡ 0 (mod a)` and `aligned ≥ off`.
///
/// **Proof**: Let `off = q·a + r` with `0 ≤ r < a`.
/// - If `r = 0`: `aligned = q·a` = off. ✓
/// - If `r > 0`: `aligned = (q+1)·a`. Since `a = 2^k`, `(q+1)·a ≡ 0 (mod a)`. ✓
/// - In both cases `aligned ≥ off`. ∎
#[allow(missing_debug_implementations)]
pub struct BumpAllocator {
    memory: NonNull<u8>,
    offset: RefCell<usize>,
    total_size: usize,
    segment_ptr: *mut Segment,
}

impl BumpAllocator {
    /// Create a bump allocator with `size_bytes` capacity (64-byte aligned).
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn new(size_bytes: usize) -> KwaversResult<Self> {
        // SAFETY: `size_bytes` is validated by mnemosyne's `is_valid_alloc_request`;
        // a null user pointer is mapped to the descriptive allocation error below.
        // mnemosyne allocates at least `SEGMENT_ALIGN` alignment (>= 64 bytes).
        let user_ptr = unsafe {
            allocate_large_or_huge::<MemoryBackendWrapper>(size_bytes, SEGMENT_ALIGN, false)
        };
        let memory = NonNull::new(user_ptr).ok_or_else(|| {
            KwaversError::System(crate::error::SystemError::MemoryAllocation {
                requested_bytes: size_bytes,
                reason: "Failed to allocate memory for bump allocator".to_owned(),
            })
        })?;

        // SAFETY: Valid pointer to allocated memory.
        // Segment pointer is stored in metadata slot by allocate_large_or_huge.
        let segment_ptr = unsafe { *((memory.as_ptr() as *mut *mut Segment).sub(1)) };

        Ok(Self {
            memory,
            offset: RefCell::new(0),
            total_size: size_bytes,
            segment_ptr,
        })
    }

    /// Allocate `size` bytes with `align`-byte alignment.
    ///
    /// Returns `KwaversError::System` when the pool is exhausted.
    /// # Errors
    /// - Returns `KwaversError::System` if the precondition for a System-class constraint is violated.
    ///
    /// # Panics
    /// - Panics if `bump allocator pointer must be non-null`.
    ///
    pub fn allocate(&self, size: usize, align: usize) -> KwaversResult<NonNull<u8>> {
        let mut offset = self.offset.borrow_mut();
        let aligned_offset = (*offset + align - 1) & !(align - 1);

        if aligned_offset + size > self.total_size {
            return Err(KwaversError::System(
                crate::error::SystemError::MemoryAllocation {
                    requested_bytes: size,
                    reason: format!(
                        "Bump allocator out of memory: {} requested, {} available",
                        size,
                        self.total_size - *offset
                    ),
                },
            ));
        }

        // SAFETY: `aligned_offset + size ≤ total_size` (checked above).
        let ptr = unsafe { self.memory.as_ptr().add(aligned_offset) };
        *offset = aligned_offset + size;

        Ok(NonNull::new(ptr).expect("bump allocator pointer must be non-null"))
    }

    /// Reset the allocator — all previous allocations are invalidated.
    pub fn reset(&self) {
        *self.offset.borrow_mut() = 0;
    }

    /// Bytes consumed since last [`reset`][Self::reset].
    pub fn used_bytes(&self) -> usize {
        *self.offset.borrow()
    }

    /// Total capacity in bytes.
    pub fn total_bytes(&self) -> usize {
        self.total_size
    }
}

impl Drop for BumpAllocator {
    fn drop(&mut self) {
        // SAFETY: segment_ptr was written by allocate_large_or_huge and is valid.
        // Recovered via the standard metadata-slot pattern.
        let _released = unsafe {
            deallocate_large_or_huge::<MemoryBackendWrapper>(self.memory.as_ptr(), self.segment_ptr)
        };
    }
}

// ─── ScopedArena ──────────────────────────────────────────────────────────────

/// A [`FieldArena`] wrapper that tracks allocated slot indices for ordered cleanup.
///
/// Useful when the caller cannot hold RAII guards returned by
/// [`FieldArena::allocate_field`] (e.g. in iterator adaptors).
#[allow(missing_debug_implementations)]
pub struct ScopedArena {
    arena: FieldArena,
    /// Ordered list of allocated field indices (for cleanup bookkeeping).
    allocated_stack: RefCell<Vec<usize>>,
}

impl ScopedArena {
    /// Create a scoped arena with the given configuration.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn new(config: ArenaConfig) -> KwaversResult<Self> {
        Ok(Self {
            arena: FieldArena::new(config)?,
            allocated_stack: RefCell::new(Vec::new()),
        })
    }

    /// Allocate a field; it will be freed when the arena is dropped.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn alloc_field(&mut self) -> KwaversResult<&mut [f64]> {
        let field = self.arena.allocate_field()?;
        // Record slot index count for deallocation bookkeeping.
        let count = self.allocated_stack.borrow().len();
        self.allocated_stack.borrow_mut().push(count);
        Ok(field)
    }

    /// Snapshot of current allocation statistics.
    pub fn stats(&self) -> ArenaStats {
        self.arena.stats()
    }
}
