//! GPU backend abstraction
//!
//! Re-exports CoreGpuContext as the primary GPU backend type.

use super::CoreGpuContext;

/// GPU backend for acoustic simulations
///
/// Type alias for CoreGpuContext for convenience.
pub type GpuBackend = CoreGpuContext;
