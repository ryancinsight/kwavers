//! Optimized Memory Layout and Field Organization
//!
//! This module provides cache-efficient memory layout strategies for simulation fields.
//! It implements Structure-of-Arrays (SoA), Array-of-Structures (AoS), and hybrid layouts
//! optimized for different access patterns and cache hierarchies.
//!
//! # Mathematical Foundation
//!
//! ## Cache Line Optimization
//!
//! For cache line size $L = 64$ bytes and element size $S = 8$ bytes (f64):
//! - Elements per cache line: $N_L = \lfloor L / S \rfloor = 8$ elements
//! - Cache line alignment: $\forall \text{ptr}, \text{ptr} \equiv 0 \pmod{64}$
//!
//! ## Temporal Locality Analysis
//!
//! Access pattern efficiency depends on striding:
//! - **Sequential access**: Stride-1, cache hits $\approx 1 - \frac{1}{N_L}$ per line
//! - **Strided access**: Stride-$s$, cache hits $\approx \frac{1}{s}$ (poor utilization)
//! - **Tiled access**: Block size $B$ matching cache, hits $\approx 1 - \frac{1}{N_L}$
//!
//! # References
//! - Wolfe M. (1989). "More Iteration Space Tiling". Supercomputing '89.
//! - Lam M.S. et al. (1991). "The Cache Performance of Blocked Algorithms". ASPLOS.
//! - Hennessy J., Patterson D. (2019). *Computer Architecture: A Quantitative Approach*, 6th Ed.

pub mod alignment;
pub mod numa_aware;
pub mod packing;
pub mod pool;
pub mod soa;
pub mod tiling;

pub use alignment::*;
pub use numa_aware::*;
pub use packing::*;
pub use pool::*;
pub use soa::*;
pub use tiling::*;
// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS AND TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Cache line size in bytes (64 bytes for x86_64)
pub const CACHE_LINE_SIZE: usize = 64;

/// Elements per cache line for f64 (64 / 8 = 8)
pub const ELEMENTS_PER_CACHE_LINE: usize = 8;

/// Preferred alignment for SIMD operations (32 bytes for AVX2)
pub const SIMD_ALIGNMENT: usize = 32;

/// NUMA-aware first-touch alignment
pub const NUMA_ALIGNMENT: usize = 4096; // Page size for first-touch

/// Memory layout type for field storage
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldLayout {
    /// Structure of Arrays: Each field stored contiguously
    /// Optimal for: Single-field operations, vectorization
    StructureOfArrays,

    /// Array of Structures: All fields stored together per element
    /// Optimal for: Multi-field point operations (e.g., gather/scatter)
    ArrayOfStructures,

    /// Hybrid layout: groups of fields use SoA within AoS blocks
    /// Optimal for: Mixed access patterns with cache blocking
    HybridSoaAos { block_size: usize },

    /// Strided layout with configurable stride
    /// Optimal for: Multithreaded access with false-sharing prevention
    Strided { stride: usize, interleave: usize },
}

/// Cache access pattern optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    /// Sequential stream access (prefetch friendly)
    Sequential,
    /// Random access (cache-oblivious tiling)
    Random,
    /// Strided access with fixed stride
    Strided(usize),
    /// Tiled/blocked access for cache reuse
    Tiled {
        tile_x: usize,
        tile_y: usize,
        tile_z: usize,
    },
}

/// NUMA memory policy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumaPolicy {
    /// No NUMA optimization
    None,
    /// First-touch: memory allocated on first accessing NUMA node
    FirstTouch,
    /// Bind to specific NUMA node
    BindToNode(usize),
    /// Interleaved across all nodes
    Interleaved,
}
