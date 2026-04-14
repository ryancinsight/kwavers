//! GPU error types

use std::fmt;

/// GPU error types with structured diagnostics
#[derive(Debug, Clone)]
pub enum GpuError {
    /// Out of GPU memory
    OutOfMemory {
        /// Requested allocation size in bytes
        requested: usize,
        /// Available memory in bytes
        available: usize,
        /// Current usage in bytes
        current: usize,
    },
    /// Device lost during operation
    DeviceLost {
        /// Ongoing operation that triggered loss
        operation: String,
    },
    /// Buffer mapping failure
    MapFailure {
        /// Buffer size in bytes
        size: usize,
        /// Error details
        details: String,
    },
    /// Validation error from wgpu
    Validation {
        /// Error message from wgpu
        message: String,
    },
    /// Timeout during GPU operation
    Timeout {
        /// Operation name
        operation: String,
        /// Timeout duration
        duration_ms: u64,
    },
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::OutOfMemory {
                requested,
                available,
                current,
            } => {
                write!(
                    f,
                    "GPU OOM: requested {} bytes, available {} bytes, current usage {} bytes",
                    requested, available, current
                )
            }
            GpuError::DeviceLost { operation } => {
                write!(f, "GPU device lost during operation: {}", operation)
            }
            GpuError::MapFailure { size, details } => {
                write!(f, "GPU buffer map failure ({} bytes): {}", size, details)
            }
            GpuError::Validation { message } => {
                write!(f, "GPU validation error: {}", message)
            }
            GpuError::Timeout {
                operation,
                duration_ms,
            } => {
                write!(
                    f,
                    "GPU operation timeout: {} ({} ms)",
                    operation, duration_ms
                )
            }
        }
    }
}

impl std::error::Error for GpuError {}
