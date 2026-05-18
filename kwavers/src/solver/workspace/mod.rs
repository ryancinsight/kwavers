//! Memory-efficient workspace for solver operations
//!
//! Provides pre-allocated workspace arrays to minimize allocations
//! during simulation, and the [`ScratchArena`] trait that unifies the shared
//! contract across all solver workspace types.
//!
//! # Design principles
//! - **SSOT**: One canonical trait for scratch-buffer memory management.
//! - **DRY**: `ScratchArena` eliminates per-solver boilerplate.
//! - **Performance**: Zero-allocation hot paths via pre-allocated buffers.
//!
//! # Invariant (Memory Monotonicity)
//! For any `T: ScratchArena`, `T::memory_bytes()` is constant after construction.
//! `T::clear()` sets all elements to zero without reallocation, so
//! `memory_bytes()` before and after `clear()` are equal.

pub mod inplace_ops;
mod pool;
mod solver_workspace;
#[cfg(test)]
mod tests;
pub use pool::{WorkspaceGuard, WorkspacePool};
pub use solver_workspace::SolverWorkspace;

/// Shared contract for pre-allocated solver scratch buffers.
///
/// Every solver workspace that allocates arrays at construction time and reuses
/// them across time-loop iterations must implement `ScratchArena` to expose a
/// uniform interface for memory reporting and buffer invalidation.
///
/// # Invariants
///
/// 1. **Memory stability** — `memory_bytes()` returns the same value throughout
///    the lifetime of the arena (construction does not change the footprint).
/// 2. **Zero after clear** — after `clear()` returns, every element of every
///    scratch buffer is exactly zero (or `Complex::zero()` for complex buffers).
/// 3. **No reallocation** — `clear()` must not allocate; it only fills existing
///    backing storage.
///
/// # Mathematical basis
///
/// Let W be a workspace allocated for an N-element grid. Define:
/// ```text
/// StaticFootprint(W) := Σ_i (|buf_i| × sizeof(elem_i))
/// ```
/// `memory_bytes()` returns `StaticFootprint(W)` in O(1).
/// `clear()` executes O(StaticFootprint(W)/cache_line) cache-line writes.
pub trait ScratchArena {
    /// Total statically pre-allocated memory in bytes.
    ///
    /// Counts only the persistent buffers allocated at construction time.
    /// Transient allocations inside solver methods are excluded.
    /// This operation is O(1).
    fn memory_bytes(&self) -> usize;

    /// Zero all scratch buffers in-place without reallocating.
    ///
    /// After this call every element of every pre-allocated buffer is `0.0`
    /// (or `Complex { re: 0.0, im: 0.0 }` for complex buffers).  The arena
    /// may be immediately reused for a subsequent simulation step.
    fn clear(&mut self);
}
