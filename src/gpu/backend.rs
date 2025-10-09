//! GPU backend abstraction
//!
//! This module provides backward-compatible type alias for GpuBackend.
//! The actual implementation is in GpuContext.

use crate::KwaversResult;
use super::GpuContext;

/// GPU backend for acoustic simulations
///
/// Type alias for GpuContext to maintain API compatibility.
/// All GPU operations should use GpuContext directly.
pub type GpuBackend = GpuContext;

/// Create GPU backend (compatibility wrapper)
///
/// This function provides backward compatibility with older API.
/// New code should use `GpuContext::new()` directly.
#[deprecated(since = "2.14.0", note = "Use GpuContext::new() directly")]
pub async fn create_gpu_backend() -> KwaversResult<GpuBackend> {
    GpuContext::new().await
}
