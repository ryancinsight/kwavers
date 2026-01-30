//! Compute Backend Abstraction Layer
//!
//! Provides abstraction for different computational backends (CPU, GPU, etc.)
//! with pluggable implementations for cross-platform simulation.

pub mod gpu;
pub mod traits;

// Re-export main types
pub use traits::{Backend, BackendCapabilities, BackendType, ComputeDevice};
