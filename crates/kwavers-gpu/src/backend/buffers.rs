//! Provider-owned GPU buffer management.
//!
//! Handles allocation, transfer, and lifecycle of GPU buffers.

use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3 as LetoArray3;
use std::collections::HashMap;
use wgpu;

/// Provider contract for backend buffer managers.
pub trait BackendBufferProvider: std::fmt::Debug {
    /// Get total allocated memory.
    fn total_allocated(&self) -> usize;

    /// Clear provider-owned buffer pool state.
    fn clear_pool(&mut self);
}

/// Provider-generic backend buffer manager.
#[derive(Debug)]
pub struct GpuBackendBufferManager<P = WgpuBackendBufferManager>
where
    P: BackendBufferProvider,
{
    provider: P,
}

impl GpuBackendBufferManager<WgpuBackendBufferManager> {
    /// Create a new WGPU buffer manager.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        WgpuBackendBufferManager::new(device).into()
    }
}

impl<P> GpuBackendBufferManager<P>
where
    P: BackendBufferProvider,
{
    /// Build a generic wrapper from a concrete buffer provider.
    #[must_use]
    pub const fn from_provider(provider: P) -> Self {
        Self { provider }
    }

    /// Borrow the concrete buffer provider.
    #[must_use]
    pub const fn provider(&self) -> &P {
        &self.provider
    }

    /// Mutably borrow the concrete buffer provider.
    #[must_use]
    pub fn provider_mut(&mut self) -> &mut P {
        &mut self.provider
    }

    /// Get total allocated memory.
    #[must_use]
    pub fn total_allocated(&self) -> usize {
        self.provider.total_allocated()
    }

    /// Clear buffer pool state.
    pub fn clear_pool(&mut self) {
        self.provider.clear_pool();
    }
}

impl<P> From<P> for GpuBackendBufferManager<P>
where
    P: BackendBufferProvider,
{
    fn from(provider: P) -> Self {
        Self::from_provider(provider)
    }
}

/// WGPU buffer manager for GPU memory.
///
/// Manages allocation, reuse, and transfer of GPU buffers.
/// Implements buffer pooling to reduce allocation overhead.
#[derive(Debug)]
pub struct WgpuBackendBufferManager {
    /// Device reference (non-owning)
    _device_ptr: *const wgpu::Device,

    /// Buffer pool for reuse
    buffer_pool: HashMap<BufferKey, Vec<wgpu::Buffer>>,

    /// Total allocated memory (bytes)
    total_allocated: usize,
}

/// Key for buffer identification and pooling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BufferKey {
    /// Buffer size in bytes
    size: usize,
    /// Buffer usage flags
    usage: u32,
}

impl BackendBufferProvider for WgpuBackendBufferManager {
    fn total_allocated(&self) -> usize {
        self.total_allocated
    }

    fn clear_pool(&mut self) {
        self.buffer_pool.clear();
    }
}

impl WgpuBackendBufferManager {
    /// Create a new buffer manager
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            _device_ptr: device as *const wgpu::Device,
            buffer_pool: HashMap::new(),
            total_allocated: 0,
        }
    }

    /// Allocate or reuse a buffer
    pub fn get_buffer(
        &mut self,
        device: &wgpu::Device,
        size: usize,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        let key = BufferKey {
            size,
            usage: usage.bits(),
        };

        // Try to reuse from pool
        if let Some(pool) = self.buffer_pool.get_mut(&key) {
            if let Some(buffer) = pool.pop() {
                return buffer;
            }
        }

        // Allocate new buffer
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kwavers-compute-buffer"),
            size: size as u64,
            usage,
            mapped_at_creation: false,
        });

        self.total_allocated += size;

        buffer
    }

    /// Return buffer to pool for reuse
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn return_buffer(&mut self, buffer: wgpu::Buffer, usage: wgpu::BufferUsages) {
        let size = buffer.size() as usize;
        let key = BufferKey {
            size,
            usage: usage.bits(),
        };

        self.buffer_pool.entry(key).or_default().push(buffer);
    }

    /// Create buffer for provider-native scalar data.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn create_buffer_from_provider_array(
        &mut self,
        device: &wgpu::Device,
        data: &LetoArray3<f32>,
        usage: wgpu::BufferUsages,
    ) -> KwaversResult<wgpu::Buffer> {
        let data_contiguous = data.as_slice().ok_or_else(|| {
            KwaversError::InvalidInput(
                "buffer_upload: provider array must be dense row-major Leto Array3".to_string(),
            )
        })?;
        let byte_data: &[u8] = bytemuck::cast_slice(data_contiguous);

        let buffer = self.get_buffer(
            device,
            byte_data.len(),
            usage | wgpu::BufferUsages::COPY_DST,
        );

        // The buffer is written by the caller via `queue.write_buffer` / a copy
        // pass; no command encoder is created here (the previous encoder was
        // allocated and immediately dropped without recording or submission).
        Ok(buffer)
    }

    /// Write provider-native scalar data into an existing buffer.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn write_provider_array_to_buffer(
        &self,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        data: &LetoArray3<f32>,
    ) -> KwaversResult<()> {
        let data_contiguous = data.as_slice().ok_or_else(|| {
            KwaversError::InvalidInput(
                "buffer_upload: provider array must be dense row-major Leto Array3".to_string(),
            )
        })?;
        let byte_data = bytemuck::cast_slice(data_contiguous);

        queue.write_buffer(buffer, 0, byte_data);

        Ok(())
    }

    /// Read buffer data into provider-native scalar storage.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub async fn read_buffer_to_provider_array(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        shape: [usize; 3],
    ) -> KwaversResult<LetoArray3<f32>> {
        self.read_buffer_to_provider_array_blocking(device, queue, buffer, shape)
    }

    fn read_buffer_to_provider_array_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        shape: [usize; 3],
    ) -> KwaversResult<LetoArray3<f32>> {
        // Create staging buffer for readback
        let size = buffer.size();
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging-buffer"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy from GPU buffer to staging buffer
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("buffer-copy"),
        });

        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);

        // Submit and wait. The submission index is not needed (we block on the
        // explicit `device.poll(Wait)` below), but `submit` is `#[must_use]` so
        // it is bound rather than dropped.
        let _submission_index = queue.submit(std::iter::once(encoder.finish()));
        queue.on_submitted_work_done(move || {
            // Work done
        });

        // Map buffer for reading
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        let _ = device.poll(wgpu::PollType::Wait);

        rx.recv()
            .map_err(|e| {
                KwaversError::GpuError(format!("buffer_map: Failed to map buffer: {}", e))
            })?
            .map_err(|e| {
                KwaversError::GpuError(format!("buffer_map: Buffer mapping error: {:?}", e))
            })?;

        // Read data
        let data_view = buffer_slice.get_mapped_range();
        let data_f32: &[f32] = bytemuck::cast_slice(&data_view);
        let data = data_f32.to_vec();

        // Reshape to Leto Array3.
        let array = LetoArray3::from_shape_vec(shape, data).map_err(|e| {
            KwaversError::GpuError(format!("shape: Failed to reshape Leto array: {}", e))
        })?;

        // Unmap buffer
        drop(data_view);
        staging_buffer.unmap();

        Ok(array)
    }

    /// Synchronous read (blocks on async)
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn read_buffer_to_provider_array_sync(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        shape: [usize; 3],
    ) -> KwaversResult<LetoArray3<f32>> {
        self.read_buffer_to_provider_array_blocking(device, queue, buffer, shape)
    }

    /// Get total allocated memory
    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }

    /// Clear buffer pool
    pub fn clear_pool(&mut self) {
        self.buffer_pool.clear();
    }
}

// Safety: WgpuBackendBufferManager is Send (device pointer is just for tracking, not dereferencing)
unsafe impl Send for WgpuBackendBufferManager {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{GpuDevice, GpuDeviceProvider};
    use hephaestus_wgpu::WgpuDevice;

    #[test]
    fn test_buffer_key() {
        let key1 = BufferKey {
            size: 1024,
            usage: 1,
        };
        let key2 = BufferKey {
            size: 1024,
            usage: 1,
        };
        let key3 = BufferKey {
            size: 2048,
            usage: 1,
        };

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_buffer_manager_creation() {
        if let Ok(provider) = GpuDevice::<WgpuDevice>::try_create_with_features_and_limits(
            WgpuDevice::acquisition_preference(),
            &[],
            WgpuDevice::required_limits(),
        ) {
            let manager = WgpuBackendBufferManager::new(provider.wgpu_device());
            assert_eq!(manager.total_allocated(), 0);
        }
    }

    #[test]
    fn backend_buffer_manager_wrapper_is_generic_over_provider_trait() {
        fn assert_provider<P>()
        where
            P: BackendBufferProvider,
        {
            let _ = core::mem::size_of::<GpuBackendBufferManager<P>>();
        }

        assert_provider::<WgpuBackendBufferManager>();
    }
}
