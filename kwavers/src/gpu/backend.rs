//! GPU backend abstraction
//!
//! Re-exports GpuContext as the primary GPU backend type.

use super::GpuContext;

/// GPU backend for acoustic simulations
///
/// Type alias for GpuContext for convenience.
pub type GpuBackend = GpuContext;
