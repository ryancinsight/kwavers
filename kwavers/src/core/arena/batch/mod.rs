//! Batch Field Allocation for Multi-Field Operations
//!
//! Provides pre-allocation strategies for simulation fields to eliminate
//! per-step allocations in hot loops and improve cache locality through
//! Structure of Arrays (SoA) layouts.
//!
//! # Mathematical Specification
//!
//! **Batch Allocation**: Given `n` fields of size `m` elements each,
//! allocate a single contiguous block `B` of size `n × m × sizeof(T)`.
//!
//! **SoA Layout**: For fields F₁, F₂, …, Fₙ:
//! ```text
//! Memory: [F₁[0..m] | F₂[0..m] | … | Fₙ[0..m]]
//! ```
//!
//! **Cache Efficiency**: Sequential access to Fᵢ achieves stride-1 pattern
//! with prefetch distance `d = cache_line_size / sizeof(T)` elements.
//!
//! # References
//!
//! - Drepper U. (2007). "What Every Programmer Should Know About Memory".
//! - Calder B. et al. (1998). "Cache-Conscious Data Placement", ASPLOS.

mod allocator;
mod config;
mod handle;
mod pool;
mod soa_buffer;
#[cfg(test)]
mod tests;

pub use allocator::BatchFieldAllocator;
pub use config::BatchFieldConfig;
pub use handle::BatchFieldHandle;
pub use pool::{BufferSize, TempBufferPool};
pub use soa_buffer::SoAFieldBuffer;

/// Cache line size in bytes (x86_64).
pub const CACHE_LINE_SIZE: usize = 64;
