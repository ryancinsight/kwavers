//! GPU backend abstraction

use crate::KwaversResult;
use crate::error::KwaversError;

/// GPU backend for acoustic simulations
#[derive(Debug)]
pub struct GpuBackend {
    // Placeholder for future GPU implementation
    _placeholder: (),
}

impl GpuBackend {
    /// Create GPU backend
    pub async fn new() -> KwaversResult<Self> {
        // Placeholder implementation
        Err(KwaversError::GpuError("GPU backend not yet fully implemented".to_string()))
    }
}
