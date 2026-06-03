//! O(1) GPU memory allocation tracking with atomic operations.
//!
//! # Mathematical Guarantee
//!
//! For device capacity M and safety factor α:
//! ```text
//! GPU_Memory_Used ≤ M × α
//! ```

pub mod config;
pub mod stats;
#[cfg(test)]
mod tests;
pub mod tracker;

pub use config::GpuAllocationConfig;
pub use stats::GpuAllocationStats;
pub use tracker::{GpuAllocationGuard, GpuAllocationTracker};
