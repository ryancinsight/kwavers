//! CUDA memory management
//!
//! This module handles memory allocation and transfers for CUDA devices.

use crate::error::{KwaversError, KwaversResult, MemoryTransferDirection};
use ndarray::Array3;

#[cfg(feature = "cudarc")]
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};

/// CUDA memory operations
pub struct CudaMemory;

impl CudaMemory {
    /// Safely get array slice, ensuring standard layout
    pub fn get_safe_slice(array: &Array3<f64>) -> KwaversResult<&[f64]> {
        if array.is_standard_layout() {
            array.as_slice().ok_or_else(|| {
                KwaversError::Gpu(crate::error::GpuError::MemoryTransfer {
                    direction: MemoryTransferDirection::HostToDevice,
                    size_bytes: array.len() * std::mem::size_of::<f64>(),
                    reason: "Failed to get array slice despite standard layout".to_string(),
                })
            })
        } else {
            Err(KwaversError::Gpu(crate::error::GpuError::MemoryTransfer {
                direction: MemoryTransferDirection::HostToDevice,
                size_bytes: array.len() * std::mem::size_of::<f64>(),
                reason: "Array is not in standard layout".to_string(),
            }))
        }
    }

    /// Safely get mutable array slice, ensuring standard layout
    pub fn get_safe_slice_mut(array: &mut Array3<f64>) -> KwaversResult<&mut [f64]> {
        if array.is_standard_layout() {
            let size_bytes = array.len() * std::mem::size_of::<f64>();
            array.as_slice_mut().ok_or_else(|| {
                KwaversError::Gpu(crate::error::GpuError::MemoryTransfer {
                    direction: MemoryTransferDirection::DeviceToHost,
                    size_bytes,
                    reason: "Failed to get mutable array slice".to_string(),
                })
            })
        } else {
            let size_bytes = array.len() * std::mem::size_of::<f64>();
            Err(KwaversError::Gpu(crate::error::GpuError::MemoryTransfer {
                direction: MemoryTransferDirection::DeviceToHost,
                size_bytes,
                reason: "Array is not in standard layout".to_string(),
            }))
        }
    }

    /// Copy array to GPU device
    #[cfg(feature = "cudarc")]
    pub fn copy_to_device(
        device: &CudaDevice,
        array: &Array3<f64>,
    ) -> KwaversResult<CudaSlice<f64>> {
        let slice = Self::get_safe_slice(array)?;

        device.htod_sync_copy(slice).map_err(|e| {
            KwaversError::Gpu(crate::error::GpuError::MemoryTransfer {
                direction: MemoryTransferDirection::HostToDevice,
                size_bytes: slice.len() * std::mem::size_of::<f64>(),
                reason: format!("CUDA htod_sync_copy failed: {:?}", e),
            })
        })
    }

    /// Copy array from GPU device
    #[cfg(feature = "cudarc")]
    pub fn copy_from_device(
        device: &CudaDevice,
        d_array: &CudaSlice<f64>,
        array: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let slice = Self::get_safe_slice_mut(array)?;

        device.dtoh_sync_copy_into(d_array, slice).map_err(|e| {
            KwaversError::Gpu(crate::error::GpuError::MemoryTransfer {
                direction: MemoryTransferDirection::DeviceToHost,
                size_bytes: slice.len() * std::mem::size_of::<f64>(),
                reason: format!("CUDA dtoh_sync_copy failed: {:?}", e),
            })
        })
    }

    /// Allocate memory on device
    #[cfg(feature = "cudarc")]
    pub fn allocate_device(device: &CudaDevice, size: usize) -> KwaversResult<CudaSlice<f64>> {
        device.alloc_zeros(size).map_err(|e| {
            KwaversError::Gpu(crate::error::GpuError::AllocationFailed {
                size_bytes: size * std::mem::size_of::<f64>(),
                reason: format!("CUDA allocation failed: {:?}", e),
            })
        })
    }
}

/// Allocate CUDA memory (standalone function for compatibility)
pub fn allocate_cuda_memory(size: usize) -> KwaversResult<usize> {
    #[cfg(feature = "cudarc")]
    {
        // Return a placeholder handle
        // Real implementation would track allocations
        Ok(size)
    }

    #[cfg(not(feature = "cudarc"))]
    {
        Err(KwaversError::NotImplemented(
            "CUDA memory allocation".to_string(),
        ))
    }
}

/// Transfer data from host to device
pub fn host_to_device_cuda(host_data: &[f64], device_buffer: usize) -> KwaversResult<()> {
    #[cfg(feature = "cudarc")]
    {
        log::debug!(
            "Transferring {} bytes to CUDA device buffer {}",
            host_data.len() * std::mem::size_of::<f64>(),
            device_buffer
        );

        // Real implementation would use device context
        Err(KwaversError::NotImplemented(
            "CUDA host to device transfer".to_string(),
        ))
    }

    #[cfg(not(feature = "cudarc"))]
    {
        Err(KwaversError::NotImplemented(
            "CUDA support not compiled".to_string(),
        ))
    }
}

/// Transfer data from device to host
pub fn device_to_host_cuda(device_buffer: usize, host_data: &mut [f64]) -> KwaversResult<()> {
    #[cfg(feature = "cudarc")]
    {
        log::debug!(
            "Transferring {} bytes from CUDA device buffer {}",
            host_data.len() * std::mem::size_of::<f64>(),
            device_buffer
        );

        // Real implementation would use device context
        Err(KwaversError::NotImplemented(
            "CUDA device to host transfer".to_string(),
        ))
    }

    #[cfg(not(feature = "cudarc"))]
    {
        Err(KwaversError::NotImplemented(
            "CUDA support not compiled".to_string(),
        ))
    }
}
