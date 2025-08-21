//! GPU and acceleration error types

use serde::{Deserialize, Serialize};
use std::error::Error as StdError;
use std::fmt;

/// GPU-related errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuError {
    /// Device not found
    DeviceNotFound,
    /// Out of memory
    OutOfMemory { requested: usize, available: usize },
    /// Kernel launch failed
    KernelLaunchFailed { kernel: String, reason: String },
    /// Memory transfer failed
    MemoryTransferFailed { direction: String, size: usize },
    /// Compilation error for GPU code
    CompilationError { source: String, error: String },
    /// Synchronization error
    SynchronizationError,
    /// Feature not supported on this GPU
    UnsupportedFeature { feature: String },
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceNotFound => write!(f, "No compatible GPU device found"),
            Self::OutOfMemory {
                requested,
                available,
            } => {
                write!(
                    f,
                    "GPU out of memory: requested {} bytes, available {} bytes",
                    requested, available
                )
            }
            Self::KernelLaunchFailed { kernel, reason } => {
                write!(f, "GPU kernel {} launch failed: {}", kernel, reason)
            }
            Self::MemoryTransferFailed { direction, size } => {
                write!(
                    f,
                    "GPU memory transfer failed ({}, {} bytes)",
                    direction, size
                )
            }
            Self::CompilationError { source, error } => {
                write!(f, "GPU code compilation failed for {}: {}", source, error)
            }
            Self::SynchronizationError => write!(f, "GPU synchronization error"),
            Self::UnsupportedFeature { feature } => {
                write!(f, "GPU feature not supported: {}", feature)
            }
        }
    }
}

impl StdError for GpuError {}
