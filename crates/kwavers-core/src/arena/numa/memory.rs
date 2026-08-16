//! NUMA memory execution, delegated to mnemosyne-heap.
//!
//! mnemosyne-heap owns the kernel memory-policy calls — `bind_to_node`
//! (`mbind(MPOL_BIND)`), `allocate_interleaved` (`mbind(MPOL_INTERLEAVE)` /
//! `VirtualAllocExNuma`), and `first_touch` (volatile page touches) — the
//! provider home this module previously duplicated. This module keeps only
//! the parallel first-touch fan-out, which stays consumer-local because
//! mnemosyne sits below moirai and cannot depend on an executor.

use moirai_parallel::{for_each_index_with, Parallel};

use super::policy::PAGE_SIZE;

/// # Safety
///
/// `ptr` must be valid for `size` bytes and remain live for the duration of
/// this call.
pub unsafe fn first_touch_memory_parallel(ptr: *mut u8, size: usize, num_threads: usize) {
    if size == 0 || num_threads == 0 {
        return;
    }

    let ptr_addr: usize = ptr as usize;
    let chunk_size = size.div_ceil(num_threads);

    for_each_index_with::<Parallel, _>(num_threads, |thread_id| {
        let start = thread_id * chunk_size;
        let end = ((start + chunk_size).min(size) / PAGE_SIZE) * PAGE_SIZE;
        if start < end {
            // SAFETY: `ptr` is valid for `size` bytes by the function contract,
            // and each worker receives a disjoint page-aligned range, so the
            // reconstructed pointer is valid for `end - start` bytes; the
            // delegate only performs volatile page-stride writes on that range.
            let chunk_ptr = (ptr_addr as *mut u8).add(start);
            unsafe { mnemosyne_heap::numa::first_touch(chunk_ptr, end - start) };
        }
    });
}
