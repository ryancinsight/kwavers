//! NUMA-aware memory binding and thread affinity (placement execution).
//!
//! The placement *vocabulary* — topology snapshots, NUMA-node identities,
//! placement hints, and the current-node query — is owned by `themis-topology`
//! and re-exported here so existing `arena::numa::*` import sites resolve
//! against the single source of truth instead of the deleted hand-rolled
//! types. The memory-policy *execution* — `mbind(2)` binding, interleaved
//! allocation, and first-touch — is owned by `mnemosyne-heap::numa` and
//! reached through it directly. This module keeps only what has no provider
//! home: thread-affinity setting, the allocator wrapper, and the parallel
//! first-touch fan-out (consumer-local because mnemosyne cannot depend on
//! moirai).

#![allow(unsafe_code)]

mod affinity;
mod allocator;
mod memory;
mod policy;
#[cfg(test)]
mod tests;
mod topology;

pub use affinity::{set_thread_affinity, ThreadAffinity};
pub use allocator::NumaAllocator;
pub use memory::first_touch_memory_parallel;
pub use policy::{CACHE_LINE_SIZE, PAGE_SIZE};
pub use topology::detect_topology;

// Re-export the themis placement vocabulary (SSOT) so `arena::numa::*` keeps
// resolving for existing callers without a hand-rolled stand-in.
pub use themis::{
    current_numa_node, try_current_numa_node, CpuTopology, NumaNodeId, PlacementHint,
};
