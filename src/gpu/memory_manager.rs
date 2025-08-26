//! GPU memory management
//!
//! This module handles GPU memory allocation and transfers,
//! following RAII and SOLID principles.

use crate::error::{GpuError, KwaversResult};
use crate::gpu::context::GpuContext;
use crate::gpu::traits::{BufferHandle, GpuBuffer, GpuMemoryOps};
use log::{debug, info, warn};
use std::collections::HashMap;
use std::sync::Arc;

/// GPU memory manager following RAII principle
pub struct GpuMemoryManager {
    context: Arc<GpuContext>,
    allocated_buffers: HashMap<usize, GpuBuffer>,
    next_buffer_id: usize,
    total_allocated: usize,
    pool_size: usize,
}

impl GpuMemoryManager {
    /// Create a new memory manager
    pub fn new(context: Arc<GpuContext>, pool_size: usize) -> KwaversResult<Self> {
        Ok(Self {
            context,
            allocated_buffers: HashMap::new(),
            next_buffer_id: 0,
            total_allocated: 0,
            pool_size,
        })
    }

    /// Get total allocated memory
    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }

    /// Get available memory
    pub fn available(&self) -> usize {
        self.pool_size.saturating_sub(self.total_allocated)
    }

    /// Check if allocation would exceed pool
    fn check_allocation(&self, size_bytes: usize) -> KwaversResult<()> {
        if self.total_allocated + size_bytes > self.pool_size {
            return Err(GpuError::OutOfMemory {
                requested: size_bytes,
                available: self.available(),
            }
            .into());
        }
        Ok(())
    }

    /// Create backend-specific buffer handle
    fn create_buffer_handle(&self, size_bytes: usize) -> KwaversResult<BufferHandle> {
        use crate::gpu::backend::GpuBackend;

        match self.context.device().backend {
            GpuBackend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    // Would call cudaMalloc
                    Ok(BufferHandle::Cuda(0))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(GpuError::BackendNotAvailable("CUDA".to_string()).into())
                }
            }
            GpuBackend::OpenCL => {
                #[cfg(feature = "opencl")]
                {
                    // Would call clCreateBuffer
                    Ok(BufferHandle::OpenCL(0))
                }
                #[cfg(not(feature = "opencl"))]
                {
                    Err(GpuError::BackendNotAvailable("OpenCL".to_string()).into())
                }
            }
            GpuBackend::WebGpu => {
                #[cfg(feature = "webgpu")]
                {
                    // Would create wgpu buffer
                    Ok(BufferHandle::WebGpu(0))
                }
                #[cfg(not(feature = "webgpu"))]
                {
                    Err(GpuError::BackendNotAvailable("WebGPU".to_string()).into())
                }
            }
        }
    }

    /// Free backend-specific buffer
    fn free_buffer_handle(&self, handle: &BufferHandle) -> KwaversResult<()> {
        match handle {
            BufferHandle::Cuda(_) => {
                #[cfg(feature = "cuda")]
                {
                    // Would call cudaFree
                    Ok(())
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Ok(())
                }
            }
            BufferHandle::OpenCL(_) => {
                #[cfg(feature = "opencl")]
                {
                    // Would call clReleaseMemObject
                    Ok(())
                }
                #[cfg(not(feature = "opencl"))]
                {
                    Ok(())
                }
            }
            BufferHandle::WebGpu(_) => {
                #[cfg(feature = "webgpu")]
                {
                    // Would drop wgpu buffer
                    Ok(())
                }
                #[cfg(not(feature = "webgpu"))]
                {
                    Ok(())
                }
            }
        }
    }
}

impl GpuMemoryOps for GpuMemoryManager {
    fn allocate(&mut self, size_bytes: usize) -> KwaversResult<GpuBuffer> {
        self.check_allocation(size_bytes)?;

        let handle = self.create_buffer_handle(size_bytes)?;
        let buffer = GpuBuffer {
            id: self.next_buffer_id,
            size_bytes,
            handle,
        };

        self.allocated_buffers.insert(buffer.id, buffer.clone());
        self.next_buffer_id += 1;
        self.total_allocated += size_bytes;

        debug!("Allocated GPU buffer {} ({} bytes)", buffer.id, size_bytes);
        Ok(buffer)
    }

    fn deallocate(&mut self, buffer: GpuBuffer) -> KwaversResult<()> {
        if let Some(stored_buffer) = self.allocated_buffers.remove(&buffer.id) {
            self.free_buffer_handle(&stored_buffer.handle)?;
            self.total_allocated = self
                .total_allocated
                .saturating_sub(stored_buffer.size_bytes);
            debug!(
                "Deallocated GPU buffer {} ({} bytes)",
                buffer.id, stored_buffer.size_bytes
            );
        } else {
            warn!("Attempted to deallocate unknown buffer {}", buffer.id);
        }
        Ok(())
    }

    fn copy_to_device(
        &self,
        host_data: &[f64],
        device_buffer: &mut GpuBuffer,
    ) -> KwaversResult<()> {
        let size_bytes = host_data.len() * std::mem::size_of::<f64>();
        if size_bytes > device_buffer.size_bytes {
            return Err(GpuError::InvalidSize {
                expected: device_buffer.size_bytes,
                actual: size_bytes,
            }
            .into());
        }

        match &device_buffer.handle {
            BufferHandle::Cuda(_) => {
                #[cfg(feature = "cuda")]
                {
                    // Would call cudaMemcpy
                    debug!("Copying {} bytes to CUDA device", size_bytes);
                    Ok(())
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(GpuError::BackendNotAvailable("CUDA".to_string()).into())
                }
            }
            BufferHandle::OpenCL(_) => {
                #[cfg(feature = "opencl")]
                {
                    // Would call clEnqueueWriteBuffer
                    debug!("Copying {} bytes to OpenCL device", size_bytes);
                    Ok(())
                }
                #[cfg(not(feature = "opencl"))]
                {
                    Err(GpuError::BackendNotAvailable("OpenCL".to_string()).into())
                }
            }
            BufferHandle::WebGpu(_) => {
                #[cfg(feature = "webgpu")]
                {
                    // Would write to wgpu buffer
                    debug!("Copying {} bytes to WebGPU device", size_bytes);
                    Ok(())
                }
                #[cfg(not(feature = "webgpu"))]
                {
                    Err(GpuError::BackendNotAvailable("WebGPU".to_string()).into())
                }
            }
        }
    }

    fn copy_from_device(
        &self,
        device_buffer: &GpuBuffer,
        host_data: &mut [f64],
    ) -> KwaversResult<()> {
        let size_bytes = host_data.len() * std::mem::size_of::<f64>();
        if size_bytes > device_buffer.size_bytes {
            return Err(GpuError::InvalidSize {
                expected: device_buffer.size_bytes,
                actual: size_bytes,
            }
            .into());
        }

        match &device_buffer.handle {
            BufferHandle::Cuda(_) => {
                #[cfg(feature = "cuda")]
                {
                    // Would call cudaMemcpy
                    debug!("Copying {} bytes from CUDA device", size_bytes);
                    Ok(())
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(GpuError::BackendNotAvailable("CUDA".to_string()).into())
                }
            }
            BufferHandle::OpenCL(_) => {
                #[cfg(feature = "opencl")]
                {
                    // Would call clEnqueueReadBuffer
                    debug!("Copying {} bytes from OpenCL device", size_bytes);
                    Ok(())
                }
                #[cfg(not(feature = "opencl"))]
                {
                    Err(GpuError::BackendNotAvailable("OpenCL".to_string()).into())
                }
            }
            BufferHandle::WebGpu(_) => {
                #[cfg(feature = "webgpu")]
                {
                    // Would read from wgpu buffer
                    debug!("Copying {} bytes from WebGPU device", size_bytes);
                    Ok(())
                }
                #[cfg(not(feature = "webgpu"))]
                {
                    Err(GpuError::BackendNotAvailable("WebGPU".to_string()).into())
                }
            }
        }
    }
}

impl Drop for GpuMemoryManager {
    fn drop(&mut self) {
        if !self.allocated_buffers.is_empty() {
            warn!(
                "GpuMemoryManager dropping with {} allocated buffers",
                self.allocated_buffers.len()
            );
            for buffer in self.allocated_buffers.values() {
                let _ = self.free_buffer_handle(&buffer.handle);
            }
        }
    }
}
