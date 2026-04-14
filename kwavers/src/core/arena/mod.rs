//! Arena Allocators — Canonical SSOT for High-Performance Memory Management
//!
//! This module is the **single source of truth** for all arena-based allocation
//! in kwavers.  `analysis::performance::arena` re-exports from here.
//!
//! # Module Map
//!
//! | Sub-module | Contents |
//! |------------|----------|
//! | [`field_arena`] | [`ArenaConfig`], [`ArenaStats`], [`FieldArena`] |
//! | [`temp_arena`] | [`BumpAllocator`], [`ScopedArena`] |
//! | [`simulation_arena`] | [`ThreadLocalArena`], [`ThreadLocalFieldGuard`] |
//! | [`layout`] | [`SoAFieldStorage`], [`FieldLayout`], [`TiledIterator3D`], NUMA-aware layout |
//! | [`batch`] | [`BatchFieldAllocator`], [`SoAFieldBuffer`], [`TempBufferPool`] |
//! | [`pool`] | [`BufferPool`], [`PooledBuffer`], [`NumaPoolManager`] |
//! | [`numa`] | [`NumaTopology`], [`NumaPolicy`], [`NumaAllocator`] |
//!
//! # Design Principles
//!
//! - **SSOT**: All arena types defined here; other modules re-export.
//! - **SRP**: Each sub-module has exactly one allocation strategy.
//! - **DIP**: Callers depend on traits/types, not on sub-module internals.
//! - **Zero-cost**: Pre-allocated pools eliminate hot-path heap overhead.
//!
//! # References
//!
//! - Hanson D.R. (1990). *Software: Practice and Experience*, 20(1), 5–12.
//! - Berger E.D. et al. (2002). *ACM SIGPLAN Notices*, 37(1), 114–124.
//! - Evans J. (2006). *BSDCan Conference*, 157–168.
//! - Bonwick J. (1994). "The Slab Allocator". *USENIX Summer Technical Conference*.

// Unsafe pointer arithmetic is documented at each use site.
#![allow(unsafe_code)]

pub mod batch;
pub mod field_arena;
pub mod layout;
pub mod numa;
pub mod pool;
pub mod simulation_arena;
pub mod temp_arena;

// ─── Re-exports ───────────────────────────────────────────────────────────────

pub use batch::{
    BatchFieldAllocator, BatchFieldConfig, BatchFieldHandle, BufferSize, SoAFieldBuffer,
    TempBufferPool,
};
pub use field_arena::{ArenaConfig, ArenaStats, FieldArena};
pub use layout::{
    align_up, cache_aligned_size, packed_struct_size, CacheBlockSize, FieldBufferGuard,
    FieldLayout, FieldPool, NumaAwareAllocator, NumaPolicy, SoAFieldStorage, TiledIterator3D,
    CACHE_LINE_SIZE, ELEMENTS_PER_CACHE_LINE,
};
pub use numa::{
    allocate_interleaved_memory, bind_memory_to_node, current_numa_node, first_touch_memory,
    first_touch_memory_parallel, set_thread_affinity, NumaAllocator, NumaTopology, ThreadAffinity,
    MAX_NUMA_NODES, PAGE_SIZE,
};
pub use pool::{
    BufferBatch, BufferPool, NumaPoolManager, PoolConfig, PoolStats, PooledBuffer,
    DEFAULT_POOL_CAPACITY,
};
pub use simulation_arena::{ThreadLocalArena, ThreadLocalFieldGuard};
pub use temp_arena::{BumpAllocator, ScopedArena};
