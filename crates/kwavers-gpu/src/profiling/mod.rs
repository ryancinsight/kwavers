// GPU Profiling and Allocation Tracking Module
//
// Provides comprehensive GPU memory tracking and profiling capabilities
// for Sprint 220: GPU Kernel Hardening.
//
// ## Mathematical Theorem
//
// **GPU Memory Invariant**: GPU_Memory_Used ≤ GPU_Memory_Total × safety_factor (0.9)
//
// **Proof**: Pre-allocation budget check before each buffer creation.
//
// ## Usage
//
// ```rust
// use kwavers::profiling::GpuAllocationTracker;
//
// let tracker = GpuAllocationTracker::new(device, 0.9)?; // 90% threshold
// let buffer = tracker.allocate_buffer(size)?;
// let peak = tracker.peak_memory();
// ```

pub mod gpu_allocator;

pub use kwavers_core::error::gpu::GpuError;
pub use gpu_allocator::{GpuAllocationConfig, GpuAllocationStats, GpuAllocationTracker};

/// Version constant for this module
pub const PROFILING_VERSION: &str = "4.0.0-sprint220";

/// Module initialization check
#[must_use]
pub fn is_initialized() -> bool {
    true // Module is always ready
}
