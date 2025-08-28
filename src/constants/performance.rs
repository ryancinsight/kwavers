//! Performance-related constants

/// Default cache line size in bytes
pub const CACHE_LINE_SIZE: usize = 64;

/// Default L1 cache size in bytes
pub const L1_CACHE_SIZE: usize = 32 * 1024;

/// Default L2 cache size in bytes
pub const L2_CACHE_SIZE: usize = 256 * 1024;

/// Default L3 cache size in bytes
pub const L3_CACHE_SIZE: usize = 8 * 1024 * 1024;

/// SIMD vector width for AVX2 (doubles)
pub const AVX2_VECTOR_WIDTH: usize = 4;

/// SIMD vector width for AVX-512 (doubles)
pub const AVX512_VECTOR_WIDTH: usize = 8;

/// Default thread pool size
pub const DEFAULT_THREAD_POOL_SIZE: usize = 4;

/// Memory alignment for SIMD
pub const SIMD_ALIGNMENT: usize = 32;