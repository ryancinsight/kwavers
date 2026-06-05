use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(target_os = "linux")]
use super::policy::MAX_NUMA_NODES;
use super::policy::PAGE_SIZE;
use super::topology::NumaTopology;
use crate::error::{KwaversError, KwaversResult};

/// # Safety
///
/// `ptr` must point to valid allocated memory of at least `size` bytes.
/// The memory must be page-aligned. No concurrent thread may access the
/// memory during binding.
/// # Errors
/// - Returns [`KwaversError::System`] if the precondition for a System-class constraint is violated.
///
#[cfg(target_os = "linux")]
pub unsafe fn bind_memory_to_node(ptr: *mut u8, size: usize, node: usize) -> KwaversResult<()> {
    const MPOL_BIND: i32 = 2;
    const MPOL_MF_STRICT: u32 = 1;

    let mut nodemask: Vec<u64> = vec![0; (MAX_NUMA_NODES + 63) / 64];
    nodemask[node / 64] |= 1u64 << (node % 64);

    let result = libc::syscall(
        libc::SYS_mbind,
        ptr,
        size,
        MPOL_BIND,
        nodemask.as_ptr(),
        MAX_NUMA_NODES,
        MPOL_MF_STRICT,
    );

    if result < 0 {
        return Err(KwaversError::System(
            crate::error::SystemError::ResourceUnavailable {
                resource: format!("Memory binding to NUMA node {} failed", node),
            },
        ));
    }

    Ok(())
}

/// # Safety
///
/// The caller must ensure that `_ptr` is a valid pointer to a memory region of at
/// least `_size` bytes that remains valid for the duration of this call.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
#[cfg(not(target_os = "linux"))]
pub unsafe fn bind_memory_to_node(_ptr: *mut u8, _size: usize, _node: usize) -> KwaversResult<()> {
    Ok(())
}
/// Allocate interleaved memory.
/// # Errors
/// - Returns [`KwaversError::System`] if the precondition for a System-class constraint is violated.
///
#[cfg(target_os = "linux")]
pub fn allocate_interleaved_memory(layout: std::alloc::Layout) -> KwaversResult<*mut u8> {
    use std::alloc::alloc;

    let ptr = unsafe { alloc(layout) };
    if ptr.is_null() {
        return Err(KwaversError::System(
            crate::error::SystemError::MemoryAllocation {
                requested_bytes: layout.size(),
                reason: "Allocation failed for interleaved memory".to_string(),
            },
        ));
    }

    const MPOL_INTERLEAVE: i32 = 3;

    unsafe {
        let topology = NumaTopology::detect();
        let mut nodemask: Vec<u64> = vec![0; (MAX_NUMA_NODES + 63) / 64];
        for node in 0..topology.node_count {
            nodemask[node / 64] |= 1u64 << (node % 64);
        }

        let _ = libc::syscall(
            libc::SYS_mbind,
            ptr,
            layout.size(),
            MPOL_INTERLEAVE,
            nodemask.as_ptr(),
            MAX_NUMA_NODES,
            0u32,
        );
    }

    Ok(ptr)
}
/// Allocate interleaved memory.
/// # Errors
/// - Returns [`KwaversError::System`] if the precondition for a System-class constraint is violated.
///
#[cfg(target_os = "windows")]
pub fn allocate_interleaved_memory(layout: std::alloc::Layout) -> KwaversResult<*mut u8> {
    mod win_numa {
        use std::ffi::c_void;
        extern "system" {
            pub fn VirtualAllocExNuma(
                hProcess: *mut c_void,
                lpAddress: *mut c_void,
                dwSize: usize,
                flAllocationType: u32,
                flProtect: u32,
                nndPreferred: u32,
            ) -> *mut c_void;
            pub fn VirtualFree(lpAddress: *mut c_void, dwSize: usize, dwFreeType: u32) -> i32;
            pub fn GetCurrentProcess() -> *mut c_void;
        }
        pub const MEM_COMMIT: u32 = 0x00001000;
        pub const MEM_RESERVE: u32 = 0x00002000;
        pub const MEM_RELEASE: u32 = 0x00008000;
        pub const PAGE_READWRITE: u32 = 0x04;
    }

    let topology = NumaTopology::detect();
    let nodes = topology.node_count;

    if nodes <= 1 {
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            return Err(KwaversError::System(
                crate::error::SystemError::MemoryAllocation {
                    requested_bytes: layout.size(),
                    reason: "Standard allocation failed".to_owned(),
                },
            ));
        }
        return Ok(ptr);
    }

    let size = layout.size();
    let chunk_size = (size / nodes).max(PAGE_SIZE);
    let process = unsafe { win_numa::GetCurrentProcess() };

    let base_ptr = unsafe {
        win_numa::VirtualAllocExNuma(
            process,
            std::ptr::null_mut(),
            size,
            win_numa::MEM_RESERVE,
            win_numa::PAGE_READWRITE,
            0,
        )
    };

    if base_ptr.is_null() {
        return Err(KwaversError::System(
            crate::error::SystemError::MemoryAllocation {
                requested_bytes: size,
                reason: "Failed to reserve interleaved NUMA region".to_owned(),
            },
        ));
    }

    let mut offset = 0usize;
    let mut current_node = 0usize;

    while offset < size {
        let commit_size = chunk_size.min(size - offset);
        let chunk_ptr = unsafe { base_ptr.add(offset) };

        let result = unsafe {
            win_numa::VirtualAllocExNuma(
                process,
                chunk_ptr,
                commit_size,
                win_numa::MEM_COMMIT,
                win_numa::PAGE_READWRITE,
                current_node as u32,
            )
        };

        if result.is_null() {
            unsafe { win_numa::VirtualFree(base_ptr, 0, win_numa::MEM_RELEASE) };
            return Err(KwaversError::System(
                crate::error::SystemError::MemoryAllocation {
                    requested_bytes: size,
                    reason: format!(
                        "Failed to commit interleaved chunk on node {}",
                        current_node
                    ),
                },
            ));
        }

        offset += commit_size;
        current_node = (current_node + 1) % nodes;
    }

    Ok(base_ptr as *mut u8)
}
/// Allocate interleaved memory.
/// # Errors
/// - Returns [`KwaversError::System`] if the precondition for a System-class constraint is violated.
///
#[cfg(not(any(target_os = "linux", target_os = "windows")))]
pub fn allocate_interleaved_memory(layout: std::alloc::Layout) -> KwaversResult<*mut u8> {
    let ptr = unsafe { std::alloc::alloc(layout) };
    if ptr.is_null() {
        return Err(KwaversError::System(
            crate::error::SystemError::MemoryAllocation {
                requested_bytes: layout.size(),
                reason: "Allocation failed".to_string(),
            },
        ));
    }
    Ok(ptr)
}

/// # Safety
///
/// `ptr` must be valid for `size` bytes. `size` should be page-aligned.
pub unsafe fn first_touch_memory(ptr: *mut u8, size: usize) {
    let mut offset = 0usize;
    while offset < size {
        let page_ptr = ptr.add(offset) as *mut AtomicUsize;
        (*page_ptr).fetch_or(0, Ordering::Relaxed);
        offset += PAGE_SIZE;
    }
}

/// # Safety
///
/// `ptr` must be valid for `size` bytes and remain live for the duration of
/// this call.
pub unsafe fn first_touch_memory_parallel(ptr: *mut u8, size: usize, num_threads: usize) {
    let ptr_addr: usize = ptr as usize;
    let chunk_size = size.div_ceil(num_threads);

    rayon::scope(|s| {
        for thread_id in 0..num_threads {
            let start = thread_id * chunk_size;
            let end = ((start + chunk_size).min(size) / PAGE_SIZE) * PAGE_SIZE;
            s.spawn(move |_| {
                if start < end {
                    let slice = unsafe {
                        std::slice::from_raw_parts_mut(
                            (ptr_addr as *mut u8).add(start),
                            end - start,
                        )
                    };
                    for i in (0..slice.len()).step_by(PAGE_SIZE) {
                        slice[i] = 0;
                    }
                }
            });
        }
    });
}
