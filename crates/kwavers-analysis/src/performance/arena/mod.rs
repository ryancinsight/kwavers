//! Arena Allocators — re-exported from the canonical SSOT at [`kwavers_core::arena`].
// This module is a pure re-export shim. All types are defined in kwavers_core::arena.
//!
//! All arena types are defined in `kwavers_core::arena`.
//! This module is a thin re-export shim so that existing import paths of the form
//! `crate::performance::arena::*` continue to compile without change.

// Unsafe pointer arithmetic lives in core::arena; allow propagation here.
#![allow(unsafe_code)]

// ─── Re-export everything from the canonical SSOT ─────────────────────────

pub use kwavers_core::arena::batch::{
    BatchFieldAllocator, BatchFieldConfig, BatchFieldHandle, BufferSize, SoAFieldBuffer,
    TempBufferPool,
};
pub use kwavers_core::arena::field_arena::{ArenaConfig, ArenaStats, FieldArena};
pub use kwavers_core::arena::layout::{
    align_up, cache_aligned_size, packed_struct_size, ArenaLayoutNumaPolicy, CacheBlockSize,
    FieldBufferGuard, FieldLayout, FieldPool, NumaAwareAllocator, SoAFieldStorage, TiledIterator3D,
    CACHE_LINE_SIZE, ELEMENTS_PER_CACHE_LINE,
};
pub use kwavers_core::arena::numa::{
    allocate_interleaved_memory, bind_memory_to_node, current_numa_node, first_touch_memory,
    first_touch_memory_parallel, set_thread_affinity, NumaAllocator, NumaTopology, ThreadAffinity,
    MAX_NUMA_NODES, PAGE_SIZE,
};
pub use kwavers_core::arena::pool::{
    BufferBatch, BufferPool, NumaPoolManager, PoolConfig, PoolStats, PooledBuffer,
    DEFAULT_POOL_CAPACITY,
};
pub use kwavers_core::arena::simulation_arena::{ThreadLocalArena, ThreadLocalFieldGuard};
pub use kwavers_core::arena::temp_arena::{BumpAllocator, ScopedArena};

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_config() {
        let config = ArenaConfig::for_3d_fields(100, 100, 100, 4);
        assert_eq!(config.field_size, 1_000_000);
        assert_eq!(config.max_fields, 4);
        assert_eq!(config.element_size, 8); // f64
        assert_eq!(config.total_memory_bytes(), 32_000_000); // 4 * 1M * 8 bytes
    }

    #[test]
    fn test_field_arena_allocation() {
        let config = ArenaConfig {
            max_fields: 2,
            field_size: 100,
            element_size: std::mem::size_of::<f64>(),
            thread_local: false,
        };

        let mut arena = FieldArena::new(config).expect("arena must allocate");

        let field1 = arena.allocate_field().expect("first slot must be free");
        assert_eq!(field1.len(), 100);

        let field2 = arena.allocate_field().expect("second slot must be free");
        assert_eq!(field2.len(), 100);

        // Third allocation must fail — all slots occupied.
        assert!(arena.allocate_field().is_err());

        let stats = arena.stats();
        assert_eq!(stats.total_fields, 2);
        assert_eq!(stats.allocated_fields, 2);
    }

    #[test]
    fn test_bump_allocator() {
        let allocator = BumpAllocator::new(1024).expect("bump allocator must allocate");

        let ptr1 = allocator.allocate(64, 8).expect("first alloc must succeed");
        let ptr2 = allocator
            .allocate(128, 16)
            .expect("second alloc must succeed");

        assert_ne!(ptr1, ptr2);
        assert_eq!(allocator.used_bytes(), 64 + 128);
        assert_eq!(allocator.total_bytes(), 1024);

        allocator.reset();
        assert_eq!(allocator.used_bytes(), 0);
    }

    #[test]
    fn test_bump_allocator_oom() {
        let allocator = BumpAllocator::new(100).expect("bump allocator must allocate");

        allocator.allocate(50, 8).expect("first alloc must succeed");
        assert!(allocator.allocate(60, 8).is_err(), "OOM must return Err");
    }
}
