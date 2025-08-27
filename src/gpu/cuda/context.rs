//! CUDA context management
//!
//! This module handles CUDA device initialization and context management.

use crate::error::{GpuError, KwaversError, KwaversResult};
use std::sync::Arc;

#[cfg(feature = "cudarc")]
use cudarc::driver::CudaDevice;

/// CUDA execution context
pub struct CudaContext {
    #[cfg(feature = "cudarc")]
    pub(crate) device: Arc<CudaDevice>,
    #[cfg(not(feature = "cudarc"))]
    _phantom: std::marker::PhantomData<()>,
}

impl CudaContext {
    /// Create new CUDA context for specified device
    pub fn new(device_id: usize) -> KwaversResult<Self> {
        #[cfg(feature = "cudarc")]
        {
            use std::panic;

            // Catch panics from CUDA library loading failures
            let result = panic::catch_unwind(|| {
                let device = CudaDevice::new(device_id).map_err(|e| {
                    KwaversError::Gpu(GpuError::DeviceInitialization {
                        device_id: device_id as u32,
                        reason: format!("Failed to create CUDA device: {:?}", e),
                    })
                })?;

                Ok(Self { device })
            });

            match result {
                Ok(context_result) => context_result,
                Err(_) => {
                    // CUDA library loading failed (panic caught)
                    Err(KwaversError::Gpu(GpuError::DeviceInitialization {
                        device_id: device_id as u32,
                        reason: "CUDA runtime library not available".to_string(),
                    }))
                }
            }
        }

        #[cfg(not(feature = "cudarc"))]
        {
            Err(KwaversError::Gpu(GpuError::BackendNotAvailable {
                backend: "CUDA".to_string(),
                reason: "CUDA support not compiled".to_string(),
            }))
        }
    }

    /// Get device ID
    pub fn device_id(&self) -> usize {
        #[cfg(feature = "cudarc")]
        {
            self.device.ordinal()
        }
        #[cfg(not(feature = "cudarc"))]
        {
            0
        }
    }

    /// Synchronize device
    pub fn synchronize(&self) -> KwaversResult<()> {
        #[cfg(feature = "cudarc")]
        {
            self.device.synchronize().map_err(|e| {
                KwaversError::Gpu(GpuError::KernelExecution {
                    kernel_name: "synchronize".to_string(),
                    reason: format!("Failed to synchronize device: {:?}", e),
                })
            })
        }

        #[cfg(not(feature = "cudarc"))]
        {
            Err(KwaversError::NotImplemented(
                "CUDA synchronization".to_string(),
            ))
        }
    }
}
