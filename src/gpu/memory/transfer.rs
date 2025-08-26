//! GPU memory transfer operations

use super::buffer::GpuBuffer;
use crate::error::{KwaversError, KwaversResult};
use crate::gpu::GpuBackend;
use std::sync::Arc;

/// GPU memory transfer mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferMode {
    /// Synchronous transfer
    Synchronous,
    /// Asynchronous transfer
    Asynchronous,
    /// Pinned memory transfer
    Pinned,
    /// Peer-to-peer transfer
    PeerToPeer,
}

/// Transfer manager for GPU memory operations
#[derive(Debug)]
pub struct TransferManager {
    backend: Arc<dyn GpuBackend>,
    mode: TransferMode,
}

impl TransferManager {
    /// Create new transfer manager
    pub fn new(backend: Arc<dyn GpuBackend>) -> Self {
        Self {
            backend,
            mode: TransferMode::Synchronous,
        }
    }

    /// Set transfer mode
    pub fn set_mode(&mut self, mode: TransferMode) {
        self.mode = mode;
    }

    /// Upload data to GPU
    pub fn upload(&self, host_data: &[u8], buffer: &mut GpuBuffer) -> KwaversResult<()> {
        if host_data.len() != buffer.size_bytes {
            return Err(KwaversError::InvalidParameter(format!(
                "Size mismatch: {} != {}",
                host_data.len(),
                buffer.size_bytes
            )));
        }

        let device_ptr = buffer
            .device_ptr
            .ok_or_else(|| KwaversError::InvalidParameter("No device pointer".to_string()))?;

        self.backend.copy_to_device(
            host_data.as_ptr() as *const _,
            device_ptr as *mut _,
            buffer.size_bytes,
        )?;

        buffer.touch();
        Ok(())
    }

    /// Download data from GPU
    pub fn download(&self, buffer: &GpuBuffer, host_data: &mut [u8]) -> KwaversResult<()> {
        if host_data.len() != buffer.size_bytes {
            return Err(KwaversError::InvalidParameter(format!(
                "Size mismatch: {} != {}",
                host_data.len(),
                buffer.size_bytes
            )));
        }

        let device_ptr = buffer
            .device_ptr
            .ok_or_else(|| KwaversError::InvalidParameter("No device pointer".to_string()))?;

        self.backend.copy_from_device(
            device_ptr as *const _,
            host_data.as_mut_ptr() as *mut _,
            buffer.size_bytes,
        )?;

        Ok(())
    }

    /// Copy between GPU buffers
    pub fn copy_device_to_device(&self, src: &GpuBuffer, dst: &mut GpuBuffer) -> KwaversResult<()> {
        if src.size_bytes != dst.size_bytes {
            return Err(KwaversError::InvalidParameter(
                "Buffer size mismatch".to_string(),
            ));
        }

        let src_ptr = src
            .device_ptr
            .ok_or_else(|| KwaversError::InvalidParameter("No source pointer".to_string()))?;
        let dst_ptr = dst
            .device_ptr
            .ok_or_else(|| KwaversError::InvalidParameter("No destination pointer".to_string()))?;

        self.backend.copy_device_to_device(
            src_ptr as *const _,
            dst_ptr as *mut _,
            src.size_bytes,
        )?;

        dst.touch();
        Ok(())
    }
}
