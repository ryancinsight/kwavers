//! NUMA-Aware Memory Allocation and Thread Affinity
//!
//! Provides portable NUMA (Non-Uniform Memory Access) support with:
//! - **First-touch policy**: Memory allocated on accessing NUMA node
//! - **Thread-to-node affinity**: Pin threads to specific NUMA nodes
//! - **Memory interleaving**: Distribute memory across nodes for bandwidth
//! - **Topology detection**: Automatic NUMA node and CPU detection
//!
//! # Mathematical Specification
//!
//! **NUMA Distance Model**: For nodes `i` and `j`, access latency:
//! ```text
//! L(i, j) = L_local          if i = j
//!           α · L_local       if i ≠ j
//! ```
//! Typically `α ∈ [1.2, 2.0]`, meaning remote access is 20–100% slower.
//!
//! **First-Touch Optimization**: Given thread `t` on node `N(t)`:
//! - Memory page `p` is bound to `N(t)` on first access by `t`
//! - Subsequent accesses by `t` incur `L_local` (optimal)
//!
//! # Platform Support
//!
//! | Platform | Features    | Implementation                    |
//! |----------|-------------|-----------------------------------|
//! | Linux    | Full        | numa.h bindings, sched_setaffinity |
//! | Windows  | Partial     | VirtualAllocExNuma, SetThreadAffinityMask |
//! | macOS    | None        | No NUMA support (uniform memory)  |
//!
//! # References
//!
//! - Linux NUMA: `man 3 numa`, `man 2 set_mempolicy`
//! - Windows NUMA: `VirtualAllocExNuma`, `GetNumaProcessorNode`
//! - Anderson T.E. et al. (1995). "Scheduler Activations", ACM TOCS 10(1):53–79
//! - Blake G. et al. (2010). "VL-NUMA: Virtualizing NUMA…", IISWC 2010

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::core::error::{KwaversError, KwaversResult};

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS AND CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

/// Maximum supported NUMA nodes.
pub const MAX_NUMA_NODES: usize = 256;

/// Cache line size in bytes (x86_64 standard).
pub const CACHE_LINE_SIZE: usize = 64;

/// Page size for memory allocation (4 KB standard).
pub const PAGE_SIZE: usize = 4096;

/// NUMA allocation policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NumaPolicy {
    /// No NUMA optimization (OS default).
    Default,
    /// First-touch: memory allocated on accessing node.
    #[default]
    FirstTouch,
    /// Bind all allocations to a specific node.
    Bind(usize),
    /// Interleave memory across all nodes.
    Interleaved,
    /// Preferred node with fallback to any node.
    Preferred(usize),
}

/// NUMA topology information.
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Number of NUMA nodes in system.
    pub node_count: usize,
    /// Total CPU cores across all nodes.
    pub total_cpus: usize,
    /// CPUs per node (approximate, may vary between nodes).
    pub cpus_per_node: usize,
    /// Distance matrix between nodes [node_count × node_count].
    pub distance_matrix: Vec<Vec<u32>>,
    /// Whether system actually has NUMA (vs single-node UMA).
    pub has_numa: bool,
}

impl Default for NumaTopology {
    fn default() -> Self {
        Self::single_node()
    }
}

impl NumaTopology {
    /// Create single-node topology (default for non-NUMA systems).
    pub fn single_node() -> Self {
        let cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        Self {
            node_count: 1,
            total_cpus: cpus,
            cpus_per_node: cpus,
            distance_matrix: vec![vec![10]],
            has_numa: false,
        }
    }

    /// Detect actual NUMA topology at runtime.
    ///
    /// - **Linux**: Reads `/sys/devices/system/node/` and `/proc/cpuinfo`
    /// - **Windows**: Uses `GetLogicalProcessorInformationEx`
    /// - **macOS**: Returns single-node (no NUMA support)
    pub fn detect() -> Self {
        #[cfg(target_os = "linux")]
        return Self::detect_linux();

        #[cfg(target_os = "windows")]
        return Self::detect_windows();

        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        Self::single_node()
    }

    /// Linux-specific NUMA detection via `/sys/devices/system/node/`.
    #[cfg(target_os = "linux")]
    fn detect_linux() -> Self {
        use std::fs;

        let node_count = fs::read_dir("/sys/devices/system/node/")
            .ok()
            .map(|dir| {
                dir.filter_map(|entry| {
                    let path = entry.ok()?.path();
                    let name = path.file_name()?.to_string_lossy().into_owned();
                    if name.starts_with("node") && name.len() > 4 {
                        name[4..].parse::<usize>().ok()
                    } else {
                        None
                    }
                })
                .count()
            })
            .filter(|c| *c > 0)
            .unwrap_or(1);

        if node_count <= 1 {
            return Self::single_node();
        }

        // Build NUMA distance matrix from `/sys/devices/system/node/nodeN/distance`.
        let mut distance_matrix = Vec::with_capacity(node_count);
        for node in 0..node_count {
            let mut row = Vec::with_capacity(node_count);
            let dist_str =
                fs::read_to_string(format!("/sys/devices/system/node/node{}/distance", node))
                    .unwrap_or_default();
            for target in 0..node_count {
                let dist = dist_str
                    .split_whitespace()
                    .nth(target)
                    .and_then(|n| n.parse::<u32>().ok())
                    .unwrap_or(if node == target { 10 } else { 20 });
                row.push(dist);
            }
            distance_matrix.push(row);
        }

        let total_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        Self {
            node_count,
            total_cpus,
            cpus_per_node: total_cpus / node_count,
            distance_matrix,
            has_numa: true,
        }
    }

    /// Windows-specific NUMA detection.
    #[cfg(target_os = "windows")]
    fn detect_windows() -> Self {
        // Windows exposes NUMA through GetLogicalProcessorInformationEx.
        // Conservative fallback to single-node; thread affinity APIs still work.
        Self::single_node()
    }

    /// Get distance between two nodes (lower = closer; local = 10 typical).
    pub fn distance(&self, from: usize, to: usize) -> u32 {
        if from >= self.node_count || to >= self.node_count {
            return 20;
        }
        self.distance_matrix[from][to]
    }

    /// Find the nearest other node to `node`.
    pub fn nearest_node(&self, node: usize) -> Option<usize> {
        if !self.has_numa || node >= self.node_count {
            return Some(0);
        }
        let mut min_dist = u32::MAX;
        let mut nearest = None;
        for other in 0..self.node_count {
            if other == node {
                continue;
            }
            let d = self.distance(node, other);
            if d < min_dist {
                min_dist = d;
                nearest = Some(other);
            }
        }
        nearest
    }

    /// Get nodes sorted by distance from `from` (closest first).
    pub fn nodes_by_distance(&self, from: usize) -> Vec<(usize, u32)> {
        if !self.has_numa || from >= self.node_count {
            return vec![(0, 10)];
        }
        let mut nodes: Vec<(usize, u32)> = (0..self.node_count)
            .map(|n| (n, self.distance(from, n)))
            .collect();
        nodes.sort_by_key(|(_, d)| *d);
        nodes
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// THREAD AFFINITY
// ═══════════════════════════════════════════════════════════════════════════

/// Thread affinity configuration.
#[derive(Debug, Clone)]
pub struct ThreadAffinity {
    /// Target NUMA node.
    pub node: Option<usize>,
    /// Specific CPU cores (overrides `node` if set).
    pub cpus: Option<Vec<usize>>,
    /// Whether to preserve existing affinity.
    pub respect_existing: bool,
}

impl ThreadAffinity {
    /// Affinity for a specific NUMA node.
    pub fn for_node(node: usize) -> Self {
        Self {
            node: Some(node),
            cpus: None,
            respect_existing: false,
        }
    }

    /// Affinity for a specific CPU set.
    pub fn for_cpus(cpus: Vec<usize>) -> Self {
        Self {
            node: None,
            cpus: Some(cpus),
            respect_existing: false,
        }
    }

    /// Unrestricted affinity (OS default).
    pub fn unrestricted() -> Self {
        Self {
            node: None,
            cpus: None,
            respect_existing: true,
        }
    }
}

/// Bind current thread to a NUMA node or CPU set.
///
/// # Platform Support
/// - **Linux**: Uses `pthread_setaffinity_np`
/// - **Windows**: Uses `SetThreadAffinityMask`
/// - **Others**: No-op (returns `Ok`)
pub fn set_thread_affinity(affinity: &ThreadAffinity) -> KwaversResult<()> {
    set_current_thread_affinity(affinity)
}

/// Get current thread's NUMA node, or `None` if undetermined.
pub fn current_numa_node() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        let cpu = unsafe { libc::sched_getcpu() };
        if cpu < 0 {
            return None;
        }
        fs::read_to_string(format!("/sys/devices/system/cpu/cpu{}/cpulist", cpu))
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
    }

    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Linux implementation of thread affinity via `sched_setaffinity`.
#[cfg(target_os = "linux")]
fn set_current_thread_affinity(affinity: &ThreadAffinity) -> KwaversResult<()> {
    use libc::{cpu_set_t, sched_setaffinity, CPU_SET, CPU_ZERO};

    unsafe {
        let mut set: cpu_set_t = std::mem::zeroed();
        let topology = NumaTopology::detect();

        if let Some(node) = affinity.node {
            let cpus_per_node = topology.cpus_per_node;
            CPU_ZERO(&mut set);
            for cpu in (node * cpus_per_node)..((node + 1) * cpus_per_node) {
                if cpu < topology.total_cpus {
                    CPU_SET(cpu, &mut set);
                }
            }
        } else if let Some(ref cpus) = affinity.cpus {
            CPU_ZERO(&mut set);
            for &cpu in cpus {
                CPU_SET(cpu, &set);
            }
        } else {
            CPU_ZERO(&mut set);
            for cpu in 0..topology.total_cpus {
                CPU_SET(cpu, &mut set);
            }
        }

        let result = sched_setaffinity(0, std::mem::size_of::<cpu_set_t>(), &set);
        if result != 0 {
            return Err(KwaversError::System(
                crate::core::error::SystemError::ResourceUnavailable {
                    resource: format!(
                        "CPU affinity binding failed: errno {}",
                        std::io::Error::last_os_error().raw_os_error().unwrap_or(-1)
                    ),
                },
            ));
        }
    }

    Ok(())
}

/// Windows implementation of thread affinity via `SetThreadAffinityMask`.
#[cfg(target_os = "windows")]
fn set_current_thread_affinity(affinity: &ThreadAffinity) -> KwaversResult<()> {
    // Declare Windows API symbols directly to avoid an external `windows_sys` dependency.
    extern "system" {
        fn GetCurrentThread() -> *mut std::ffi::c_void;
        fn SetThreadAffinityMask(
            hThread: *mut std::ffi::c_void,
            dwThreadAffinityMask: usize,
        ) -> usize;
    }

    unsafe {
        let topology = NumaTopology::detect();
        let mask = if let Some(node) = affinity.node {
            let cpus_per_node = topology.cpus_per_node;
            let start_cpu = node * cpus_per_node;
            ((1usize << cpus_per_node) - 1) << start_cpu
        } else {
            !0usize
        };

        let handle = GetCurrentThread();
        let old_mask = SetThreadAffinityMask(handle, mask);

        if old_mask == 0 {
            return Err(KwaversError::System(
                crate::core::error::SystemError::ResourceUnavailable {
                    resource: "Failed to set thread affinity mask".to_string(),
                },
            ));
        }
    }

    Ok(())
}

/// Other platforms — no-op.
#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn set_current_thread_affinity(_affinity: &ThreadAffinity) -> KwaversResult<()> {
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// MEMORY BINDING
// ═══════════════════════════════════════════════════════════════════════════

/// Bind memory region to a NUMA node using `mbind(2)` (Linux only).
///
/// # Safety
///
/// - `ptr` must point to valid allocated memory of at least `size` bytes.
/// - The memory must be page-aligned.
/// - No concurrent thread may access the memory during binding.
///
/// # Platform Notes
///
/// - **Linux**: Uses `SYS_mbind` with `MPOL_BIND | MPOL_MF_STRICT`.
/// - **Windows / Others**: No-op (memory binding is done at allocation time
///   via `VirtualAllocExNuma`).
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
            crate::core::error::SystemError::ResourceUnavailable {
                resource: format!("Memory binding to NUMA node {} failed", node),
            },
        ));
    }

    Ok(())
}

/// Windows / other — no-op (binding is handled by `VirtualAllocExNuma` at allocation).
///
/// # Safety
///
/// The caller must ensure that `_ptr` is a valid pointer to a memory region of at
/// least `_size` bytes that remains valid for the duration of this call.
#[cfg(not(target_os = "linux"))]
pub unsafe fn bind_memory_to_node(_ptr: *mut u8, _size: usize, _node: usize) -> KwaversResult<()> {
    Ok(())
}

/// Allocate memory with NUMA interleaving across all nodes (Linux).
///
/// Allocates normally then applies `MPOL_INTERLEAVE` via `mbind(2)`.
#[cfg(target_os = "linux")]
pub fn allocate_interleaved_memory(layout: std::alloc::Layout) -> KwaversResult<*mut u8> {
    use std::alloc::alloc;

    let ptr = unsafe { alloc(layout) };
    if ptr.is_null() {
        return Err(KwaversError::System(
            crate::core::error::SystemError::MemoryAllocation {
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

        // Failure to set policy is non-fatal (allocation already succeeded).
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

/// Windows interleaved allocation: commits pages round-robin across NUMA nodes.
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
                crate::core::error::SystemError::MemoryAllocation {
                    requested_bytes: layout.size(),
                    reason: "Standard allocation failed".to_string(),
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
            crate::core::error::SystemError::MemoryAllocation {
                requested_bytes: size,
                reason: "Failed to reserve interleaved NUMA region".to_string(),
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
                crate::core::error::SystemError::MemoryAllocation {
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

/// Fallback for other platforms — standard allocation.
#[cfg(not(any(target_os = "linux", target_os = "windows")))]
pub fn allocate_interleaved_memory(layout: std::alloc::Layout) -> KwaversResult<*mut u8> {
    let ptr = unsafe { std::alloc::alloc(layout) };
    if ptr.is_null() {
        return Err(KwaversError::System(
            crate::core::error::SystemError::MemoryAllocation {
                requested_bytes: layout.size(),
                reason: "Allocation failed".to_string(),
            },
        ));
    }
    Ok(ptr)
}

// ═══════════════════════════════════════════════════════════════════════════
// FIRST-TOUCH UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

/// Perform first-touch initialization on a memory region.
///
/// Writes to each page so it is physically allocated on the NUMA node of the
/// current thread.
///
/// # Safety
///
/// - `ptr` must be valid for `size` bytes.
/// - `size` should be page-aligned for best results.
pub unsafe fn first_touch_memory(ptr: *mut u8, size: usize) {
    let mut offset = 0usize;
    while offset < size {
        let page_ptr = ptr.add(offset) as *mut AtomicUsize;
        (*page_ptr).fetch_or(0, Ordering::Relaxed);
        offset += PAGE_SIZE;
    }
}

/// Parallel first-touch initialization using rayon.
///
/// Distributes memory across threads so each thread touches its portion,
/// establishing NUMA affinity across multiple nodes.
///
/// # Safety
///
/// `ptr` must be valid for `size` bytes and remain live for the duration of
/// this call.
pub unsafe fn first_touch_memory_parallel(ptr: *mut u8, size: usize, num_threads: usize) {
    // Encode the pointer as a plain integer so the spawned tasks capture only
    // `Send + Sync` types (`usize`).  The pointer itself is never stored in any
    // closure — it is reconstructed from the integer inside each task body.
    //
    // SAFETY: caller guarantees `ptr` is valid for `size` bytes and lives
    // through the entire parallel section; chunks are non-overlapping.
    let ptr_addr: usize = ptr as usize;
    let chunk_size = size.div_ceil(num_threads);

    rayon::scope(|s| {
        for thread_id in 0..num_threads {
            let start = thread_id * chunk_size;
            let end = ((start + chunk_size).min(size) / PAGE_SIZE) * PAGE_SIZE;
            // `move` captures only usize values — all Send + Sync.
            s.spawn(move |_| {
                if start < end {
                    // SAFETY: non-overlapping chunk within [ptr, ptr+size).
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

// ═══════════════════════════════════════════════════════════════════════════
// NUMA-AWARE ALLOCATOR WRAPPER
// ═══════════════════════════════════════════════════════════════════════════

/// NUMA-aware memory allocator wrapper.
#[derive(Debug, Clone)]
pub struct NumaAllocator {
    policy: NumaPolicy,
    topology: NumaTopology,
}

impl NumaAllocator {
    /// Create allocator with default topology detection.
    pub fn new() -> Self {
        let topology = NumaTopology::detect();
        Self {
            policy: NumaPolicy::FirstTouch,
            topology,
        }
    }

    /// Create with a specific policy.
    pub fn with_policy(policy: NumaPolicy) -> Self {
        let topology = NumaTopology::detect();
        Self { policy, topology }
    }

    /// Get current policy.
    pub fn policy(&self) -> NumaPolicy {
        self.policy
    }

    /// Get topology reference.
    pub fn topology(&self) -> &NumaTopology {
        &self.topology
    }
}

impl Default for NumaAllocator {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numa_topology_detection_sanity() {
        let topo = NumaTopology::detect();
        assert!(topo.node_count >= 1);
        assert!(topo.total_cpus >= 1);
        assert_eq!(topo.distance_matrix.len(), topo.node_count);

        for i in 0..topo.node_count {
            let local = topo.distance(i, i);
            assert!(
                local <= 20,
                "Local access distance should be ≤20, got {}",
                local
            );
        }
    }

    #[test]
    fn test_nodes_by_distance_sorted() {
        let topo = NumaTopology::detect();
        for node in 0..topo.node_count {
            let ordered = topo.nodes_by_distance(node);
            assert_eq!(ordered.len(), topo.node_count);
            assert_eq!(ordered[0].0, node, "self should be closest");
            for i in 1..ordered.len() {
                assert!(
                    ordered[i].1 >= ordered[i - 1].1,
                    "distances must be non-decreasing"
                );
            }
        }
    }

    #[test]
    fn test_thread_affinity_construction() {
        let unres = ThreadAffinity::unrestricted();
        assert!(unres.node.is_none());
        assert!(unres.cpus.is_none());

        let node = ThreadAffinity::for_node(0);
        assert_eq!(node.node, Some(0));

        let cpus = ThreadAffinity::for_cpus(vec![0, 2, 4]);
        assert_eq!(cpus.cpus, Some(vec![0, 2, 4]));
    }

    #[test]
    fn test_first_touch_memory() {
        let layout = std::alloc::Layout::from_size_align(4096, 4096).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };
        if !ptr.is_null() {
            unsafe { first_touch_memory(ptr, 4096) };
            unsafe { std::alloc::dealloc(ptr, layout) };
        }
    }

    #[test]
    fn test_numa_allocator_default() {
        let alloc = NumaAllocator::new();
        assert!(alloc.topology().node_count >= 1);
    }
}
