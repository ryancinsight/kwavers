// Memory tracking and instrumentation for solver operations
//
// This module provides comprehensive memory allocation tracking that enables:
// - Real-time transient allocation monitoring
// - Peak memory detection with O(1) overhead
// - Allocation tracing with causal chain preservation
// - Memory budget enforcement
//
// ## Mathematical Specification
//
// ### Memory Invariant Theorem
// For a simulation S with n timesteps and workspace W:
//
// ```text
// TotalMemory(S) = StaticMemory(W) + Σ TransientMemory(t) for t ∈ [1, n]
//
// where:
// - StaticMemory(W) = Σ (buffer_size × element_size) for all pre-allocated buffers
// - TransientMemory(t) = Σ (alloc_size) for all allocations at timestep t
// - PeakMemory = StaticMemory + max(TransientMemory(t)) ∀t ∈ [1, n]
// ```
//
// ### Complexity Analysis
// - allocate(): O(1) amortized
// - record_transient(): O(1) with atomic operations
// - peak_memory(): O(1) read of cached maximum
// - total_memory(): O(1) arithmetic
//
// ## References
//
// - Wilson et al. (1995) "Dynamic Memory Management"
//   ISBN: 0-201-52992-9
// - Berger et al. (2001) "Composing High-Performance Memory Allocators"
//   DOI: 10.1145/378993.379433
// - Drepper (2007) "What Every Programmer Should Know About Memory"
//   https://akkadia.org/drepper/cpumemory.pdf

mod global;
mod guard;
mod thread;

#[cfg(test)]
mod tests;

pub use global::GlobalAllocationTracker;
pub use guard::AllocationGuard;
pub use thread::ThreadAllocationTracker;
