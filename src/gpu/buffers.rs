//! GPU buffer management

use crate::domain::core::error::{KwaversError, KwaversResult};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

/// GPU buffer wrapper
#[derive(Debug)]
pub struct GpuBuffer {
    buffer: wgpu::Buffer,
    size: u64,
    usage: wgpu::BufferUsages,
    _label: String,
}

impl GpuBuffer {
    /// Create a new GPU buffer
    pub fn new(device: &wgpu::Device, label: &str, size: u64, usage: wgpu::BufferUsages) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            size,
            usage,
            _label: label.to_string(),
        }
    }

    /// Create buffer with initial data
    pub fn with_data<T: bytemuck::Pod>(
        device: &wgpu::Device,
        label: &str,
        data: &[T],
        usage: wgpu::BufferUsages,
    ) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage,
        });

        let size = std::mem::size_of_val(data) as u64;

        Self {
            buffer,
            size,
            usage,
            _label: label.to_string(),
        }
    }

    /// Get buffer reference
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Get buffer size
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Map buffer for reading
    pub async fn read<T: bytemuck::Pod>(&self) -> KwaversResult<Vec<T>> {
        if !self.usage.contains(wgpu::BufferUsages::MAP_READ) {
            return Err(KwaversError::System(
                crate::domain::core::error::SystemError::InvalidOperation {
                    operation: "Buffer reading".to_string(),
                    reason: "Buffer not created with MAP_READ usage".to_string(),
                },
            ));
        }

        let buffer_slice = self.buffer.slice(..);
        let (tx, rx) = flume::bounded(1);

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        // Wait for mapping to complete
        let result = rx.recv_async().await.map_err(|e| {
            KwaversError::System(crate::domain::core::error::SystemError::InvalidOperation {
                operation: "Buffer mapping channel".to_string(),
                reason: format!("Failed: {}", e),
            })
        })?;

        result?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        self.buffer.unmap();

        Ok(result)
    }

    /// Write data to buffer
    pub fn write<T: bytemuck::Pod>(&self, queue: &wgpu::Queue, data: &[T]) {
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(data));
    }
}

/// Buffer manager for efficient GPU memory management
#[derive(Debug)]
pub struct BufferManager {
    buffers: HashMap<String, GpuBuffer>,
    total_memory: u64,
    _max_memory: u64,
}

impl BufferManager {
    /// Create a new buffer manager
    pub fn new(device: &wgpu::Device) -> Self {
        let limits = device.limits();
        Self {
            buffers: HashMap::new(),
            total_memory: 0,
            _max_memory: limits.max_buffer_size,
        }
    }

    /// Allocate a new buffer
    pub fn allocate(
        &mut self,
        device: &wgpu::Device,
        name: &str,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> KwaversResult<&GpuBuffer> {
        if self.buffers.contains_key(name) {
            return Err(KwaversError::System(
                crate::domain::core::error::SystemError::InvalidOperation {
                    operation: format!("Buffer '{}' creation", name),
                    reason: "Buffer already exists".to_string(),
                },
            ));
        }

        let buffer = GpuBuffer::new(device, name, size, usage);
        self.total_memory += size;
        self.buffers.insert(name.to_string(), buffer);

        self.buffers.get(name).ok_or_else(|| {
            crate::domain::core::error::KwaversError::System(
                crate::domain::core::error::SystemError::ResourceExhausted {
                    resource: format!("GPU buffer '{}'", name),
                    reason: "Buffer not found after creation".to_string(),
                },
            )
        })
    }

    /// Get buffer by name
    pub fn get(&self, name: &str) -> Option<&GpuBuffer> {
        self.buffers.get(name)
    }

    /// Get mutable buffer by name
    pub fn get_mut(&mut self, name: &str) -> Option<&mut GpuBuffer> {
        self.buffers.get_mut(name)
    }

    /// Release a buffer
    pub fn release(&mut self, name: &str) {
        if let Some(buffer) = self.buffers.remove(name) {
            self.total_memory -= buffer.size();
        }
    }

    /// Get total allocated memory
    pub fn total_memory(&self) -> u64 {
        self.total_memory
    }
}
