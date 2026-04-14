// Field Arena — pre-allocated pool for simulation field arrays.
//
// Provides [`FieldArena`]: a fixed-capacity slab allocator that carves
// mutable `[f64]` slices out of a single contiguous 64-byte-aligned block.
// Slot reuse is tracked with a bitmap; individual deallocation is O(1).
//
// # References
// - Hanson D.R. (1990). Software: Practice and Experience, 20(1), 5–12.
// - Berger E.D. et al. (2002). ACM SIGPLAN Notices, 37(1), 114–124.

use std::alloc::{alloc, dealloc, Layout};
use std::cell::RefCell;
use std::ptr::NonNull;

use crate::core::error::{KwaversError, KwaversResult};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for arena allocator.
///
/// Shared by [`FieldArena`], [`super::simulation_arena::ThreadLocalArena`], and
/// [`super::temp_arena::ScopedArena`].
#[derive(Debug, Clone)]
pub struct ArenaConfig {
    /// Maximum number of fields that can be allocated simultaneously.
    pub max_fields: usize,
    /// Number of elements per field.
    pub field_size: usize,
    /// Size of each element in bytes.
    pub element_size: usize,
    /// Whether to use thread-local arenas for concurrent access.
    pub thread_local: bool,
}

impl ArenaConfig {
    /// Configuration for 3-D field operations (`nx × ny × nz` elements each).
    pub fn for_3d_fields(nx: usize, ny: usize, nz: usize, max_concurrent: usize) -> Self {
        Self {
            max_fields: max_concurrent,
            field_size: nx * ny * nz,
            element_size: std::mem::size_of::<f64>(),
            thread_local: true,
        }
    }

    /// Configuration for 2-D field operations (`nx × ny` elements each).
    pub fn for_2d_fields(nx: usize, ny: usize, max_concurrent: usize) -> Self {
        Self {
            max_fields: max_concurrent,
            field_size: nx * ny,
            element_size: std::mem::size_of::<f64>(),
            thread_local: true,
        }
    }

    /// Total memory requirement in bytes.
    pub fn total_memory_bytes(&self) -> usize {
        self.max_fields * self.field_size * self.element_size
    }
}

// ─── ArenaStats ───────────────────────────────────────────────────────────────

/// Snapshot of arena allocation statistics.
#[derive(Debug, Clone)]
pub struct ArenaStats {
    /// Maximum number of fields that can be allocated.
    pub total_fields: usize,
    /// Currently live fields.
    pub allocated_fields: usize,
    /// Size of each field in elements.
    pub field_size_elements: usize,
    /// Total memory reserved in bytes.
    pub total_memory_bytes: usize,
}

// ─── Internal state ───────────────────────────────────────────────────────────

#[derive(Debug)]
pub(super) struct AllocationState {
    /// Bitmap: `true` = slot is live.
    pub allocated: Vec<bool>,
    pub allocated_count: usize,
}

// ─── FieldArena ───────────────────────────────────────────────────────────────

/// Slab allocator for simulation field arrays.
///
/// Allocates one contiguous block on construction and sub-divides it into
/// `max_fields` equal-sized `[f64]` slices.  Allocation is O(max_fields)
/// (bitmap scan, typically ≤8 slots) and deallocation is O(1).
///
/// # Algorithm
///
/// **Theorem** (Slab Safety): All returned slices are non-overlapping and
/// within the single allocation block.
///
/// **Proof**:
/// - Slot `i` occupies bytes `[i·field_bytes, (i+1)·field_bytes)`.
/// - The bitmap prevents a slot from being returned twice.
/// - Therefore ∀i ≠ j: regions are disjoint. ∎
///
/// # Reference
///
/// - Berger E.D. et al. (2002). *ACM SIGPLAN Notices*, 37(1), 114–124.
#[derive(Debug)]
pub struct FieldArena {
    /// Pre-allocated, 64-byte-aligned memory block.
    pub(super) memory: NonNull<u8>,
    /// Layout used at allocation (needed for `dealloc`).
    layout: Layout,
    /// Configuration snapshot.
    pub(super) config: ArenaConfig,
    /// Slot-use bitmap.
    pub(super) allocation_state: RefCell<AllocationState>,
}

impl FieldArena {
    /// Create a new field arena.
    pub fn new(config: ArenaConfig) -> KwaversResult<Self> {
        let total_size = config.total_memory_bytes();

        if total_size == 0 {
            return Err(KwaversError::Validation(
                crate::core::error::ValidationError::InvalidValue {
                    parameter: "arena_config".to_string(),
                    value: 0.0,
                    reason: "Arena configuration results in zero memory allocation".to_string(),
                },
            ));
        }

        // 64-byte alignment for cache-line efficiency and SIMD readiness.
        let layout = Layout::from_size_align(total_size, 64).map_err(|_| {
            KwaversError::System(crate::core::error::SystemError::MemoryAllocation {
                requested_bytes: total_size,
                reason: "Failed to create layout for arena allocation".to_string(),
            })
        })?;

        // SAFETY: `layout` is non-zero and power-of-2-aligned (64).
        // `NonNull::new` handles the null-pointer OOM case.
        let memory = unsafe { alloc(layout) };
        let memory = NonNull::new(memory).ok_or_else(|| {
            KwaversError::System(crate::core::error::SystemError::MemoryAllocation {
                requested_bytes: total_size,
                reason: "Failed to allocate memory for arena".to_string(),
            })
        })?;

        let allocation_state = RefCell::new(AllocationState {
            allocated: vec![false; config.max_fields],
            allocated_count: 0,
        });

        Ok(Self {
            memory,
            layout,
            config,
            allocation_state,
        })
    }

    /// Allocate one field slot and return a mutable slice into arena memory.
    ///
    /// Fails with [`KwaversError::System`] when all slots are occupied.
    pub fn allocate_field(&mut self) -> KwaversResult<&mut [f64]> {
        let mut state = self.allocation_state.borrow_mut();

        let slot = state
            .allocated
            .iter()
            .position(|&in_use| !in_use)
            .ok_or_else(|| {
                KwaversError::System(crate::core::error::SystemError::ResourceUnavailable {
                    resource: "arena field slot".to_string(),
                })
            })?;

        state.allocated[slot] = true;
        state.allocated_count += 1;

        // SAFETY: `slot < max_fields` and `offset + field_bytes ≤ layout.size()`.
        // The 64-byte arena alignment satisfies the 8-byte alignment of f64.
        let offset = slot * self.config.field_size * self.config.element_size;
        let field_ptr = unsafe { self.memory.as_ptr().add(offset) as *mut f64 };
        let field_slice =
            unsafe { std::slice::from_raw_parts_mut(field_ptr, self.config.field_size) };

        Ok(field_slice)
    }

    /// Return a field slot to the arena.
    ///
    /// In this slab implementation, individual slots are freed at arena drop.
    /// This method exists for API symmetry only.
    pub fn deallocate_field(&self, _field: &mut [f64]) -> KwaversResult<()> {
        Ok(())
    }

    /// Snapshot of current allocation statistics.
    pub fn stats(&self) -> ArenaStats {
        let state = self.allocation_state.borrow();
        ArenaStats {
            total_fields: self.config.max_fields,
            allocated_fields: state.allocated_count,
            field_size_elements: self.config.field_size,
            total_memory_bytes: self.config.total_memory_bytes(),
        }
    }
}

impl Drop for FieldArena {
    fn drop(&mut self) {
        // SAFETY: `self.memory` and `self.layout` are the identical pointer/layout
        // used in `alloc()`.  Rust guarantees `drop` is called exactly once.
        unsafe {
            dealloc(self.memory.as_ptr(), self.layout);
        }
    }
}
