//! GPU Buffer Management
//!
//! Handles allocation, transfer, and lifecycle of GPU buffers.

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;
use std::collections::HashMap;
use wgpu;

/// Buffer manager for GPU memory
///
/// Manages allocation, reuse, and transfer of GPU buffers.
/// Implements buffer pooling to reduce allocation overhead.
#[derive(Debug)]
pub struct BufferManager {
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

impl BufferManager {
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
    pub fn return_buffer(&mut self, buffer: wgpu::Buffer, usage: wgpu::BufferUsages) {
        let size = buffer.size() as usize;
        let key = BufferKey {
            size,
            usage: usage.bits(),
        };

        self.buffer_pool.entry(key).or_default().push(buffer);
    }

    /// Create buffer from Array3<f64> data
    pub fn create_buffer_from_array(
        &mut self,
        device: &wgpu::Device,
        data: &Array3<f64>,
        usage: wgpu::BufferUsages,
    ) -> KwaversResult<wgpu::Buffer> {
        // Convert f64 to f32 for GPU (most GPUs don't support f64 efficiently)
        let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
        let byte_data = bytemuck::cast_slice(&data_f32);

        let buffer = self.get_buffer(
            device,
            byte_data.len(),
            usage | wgpu::BufferUsages::COPY_DST,
        );

        // Write data to buffer
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("buffer-write"),
        });

        Ok(buffer)
    }

    /// Create buffer from Array3<f64> using queue write
    pub fn write_array_to_buffer(
        &self,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        data: &Array3<f64>,
    ) -> KwaversResult<()> {
        // Convert f64 to f32
        let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
        let byte_data = bytemuck::cast_slice(&data_f32);

        queue.write_buffer(buffer, 0, byte_data);

        Ok(())
    }

    /// Read buffer data into Array3<f64>
    pub async fn read_buffer_to_array(
        &self,
        device: &wgpu::Device,
        buffer: &wgpu::Buffer,
        shape: (usize, usize, usize),
    ) -> KwaversResult<Array3<f64>> {
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

        // Submit and wait
        let submission_index = device.queue().submit(std::iter::once(encoder.finish()));
        device.queue().on_submitted_work_done(move || {
            // Work done
        });

        // Map buffer for reading
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|e| {
                KwaversError::ConfigError(crate::core::error::ConfigError::InvalidParameter {
                    param_name: "buffer_map".to_string(),
                    reason: format!("Failed to map buffer: {}", e),
                })
            })?
            .map_err(|e| {
                KwaversError::ConfigError(crate::core::error::ConfigError::InvalidParameter {
                    param_name: "buffer_map".to_string(),
                    reason: format!("Buffer mapping error: {:?}", e),
                })
            })?;

        // Read data
        let data_view = buffer_slice.get_mapped_range();
        let data_f32: &[f32] = bytemuck::cast_slice(&data_view);

        // Convert f32 back to f64
        let data_f64: Vec<f64> = data_f32.iter().map(|&x| x as f64).collect();

        // Reshape to Array3
        let array = Array3::from_shape_vec(shape, data_f64).map_err(|e| {
            KwaversError::ConfigError(crate::core::error::ConfigError::InvalidParameter {
                param_name: "shape".to_string(),
                reason: format!("Failed to reshape array: {}", e),
            })
        })?;

        // Unmap buffer
        drop(data_view);
        staging_buffer.unmap();

        Ok(array)
    }

    /// Synchronous read (blocks on async)
    pub fn read_buffer_to_array_sync(
        &self,
        device: &wgpu::Device,
        buffer: &wgpu::Buffer,
        shape: (usize, usize, usize),
    ) -> KwaversResult<Array3<f64>> {
        pollster::block_on(self.read_buffer_to_array(device, buffer, shape))
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

// Safety: BufferManager is Send (device pointer is just for tracking, not dereferencing)
unsafe impl Send for BufferManager {}

#[cfg(test)]
mod tests {
    use super::*;

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
        // Need actual device for full test
        // This is a minimal smoke test
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Try to get adapter
        if let Some(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: None,
                force_fallback_adapter: false,
            }))
        {
            if let Ok((device, _queue)) = pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("test-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )) {
                let manager = BufferManager::new(&device);
                assert_eq!(manager.total_allocated(), 0);
            }
        }
    }
}
