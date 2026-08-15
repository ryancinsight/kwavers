//! NUMA-aware memory binding and thread affinity (placement execution).
//!
//! The placement *vocabulary* — topology snapshots, NUMA-node identities,
//! placement hints, and the current-node query — is owned by `themis-topology`
//! and re-exported here so existing `arena::numa::*` import sites resolve
//! against the single source of truth instead of the deleted hand-rolled
//! types. This module keeps only the *execution* primitives themis
//! deliberately does not own (it is placement vocabulary, not allocation or
//! execution): `mbind(2)` / `VirtualAllocExNuma` binding, interleaved
//! allocation, first-touch, and thread-affinity setting.

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
pub use memory::{
    allocate_interleaved_memory, bind_memory_to_node, first_touch_memory,
    first_touch_memory_parallel,
};
pub use policy::{CACHE_LINE_SIZE, MAX_NUMA_NODES, PAGE_SIZE};
pub use topology::detect_topology;

// Re-export the themis placement vocabulary (SSOT) so `arena::numa::*` keeps
// resolving for existing callers without a hand-rolled stand-in.
pub use themis::{
    current_numa_node, try_current_numa_node, CpuTopology, NumaNodeId, PlacementHint,
};
