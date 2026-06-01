/// Maximum supported NUMA nodes.
pub const MAX_NUMA_NODES: usize = 256;

/// Cache line size in bytes (x86_64 standard).
pub const CACHE_LINE_SIZE: usize = 64;

/// Page size for memory allocation (4 KB standard).
pub const PAGE_SIZE: usize = 4096;

/// NUMA allocation policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NumaAllocPolicy {
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
