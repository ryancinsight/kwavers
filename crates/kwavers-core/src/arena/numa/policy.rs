/// Maximum supported NUMA nodes (nodemask width for `mbind`).
pub const MAX_NUMA_NODES: usize = 256;

/// Cache line size in bytes (x86_64 standard).
pub const CACHE_LINE_SIZE: usize = 64;

/// Page size for memory allocation (4 KB standard).
pub const PAGE_SIZE: usize = 4096;
