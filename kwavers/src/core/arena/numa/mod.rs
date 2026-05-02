//! NUMA-Aware Memory Allocation and Thread Affinity

#![allow(unsafe_code)]

mod affinity;
mod allocator;
mod memory;
mod policy;
#[cfg(test)]
mod tests;
mod topology;

pub use affinity::{current_numa_node, set_thread_affinity, ThreadAffinity};
pub use allocator::NumaAllocator;
pub use memory::{
    allocate_interleaved_memory, bind_memory_to_node, first_touch_memory,
    first_touch_memory_parallel,
};
pub use policy::{NumaPolicy, CACHE_LINE_SIZE, MAX_NUMA_NODES, PAGE_SIZE};
pub use topology::NumaTopology;
