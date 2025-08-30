//! GPU buffer management

use crate::error::{KwaversError, KwaversResult};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

/// GPU buffer wrapper
#[derive(Debug)]
pub struct GpuBuffer {
    buffer: wgpu::Buffer,
    size: u64,
    usage: wgpu::BufferUsages,
    label: String,
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
            label: label.to_string(),
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

        let size = (data.len() * std::mem::size_of::<T>()) as u64;

        Self {
            buffer,
            size,
            usage,
            label: label.to_string(),
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
                crate::error::SystemError::InvalidOperation {
                    operation: "Buffer not created with MAP_READ usage".to_string(),
                },
            ));
        }

        let buffer_slice = self.buffer.slice(..);
        let (tx, rx) = flume::bounded(1);

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        // Wait for mapping to complete
        rx.recv_async().await.map_err(|e| {
            KwaversError::System(crate::error::SystemError::InvalidOperation {
                operation: format!("Buffer mapping failed: {}", e),
            })
        })??;

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
}

impl BufferManager {
    /// Create a new buffer manager
    pub fn new(_device: &wgpu::Device) -> Self {
        Self {
            buffers: HashMap::new(),
            total_memory: 0,
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
                crate::error::SystemError::InvalidOperation {
                    operation: format!("Buffer '{}' already exists", name),
                },
            ));
        }

        let buffer = GpuBuffer::new(device, name, size, usage);
        self.total_memory += size;
        self.buffers.insert(name.to_string(), buffer);

        Ok(self.buffers.get(name).unwrap())
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
