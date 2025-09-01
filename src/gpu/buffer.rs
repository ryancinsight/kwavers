//! GPU buffer management

use crate::KwaversResult;
use std::marker::PhantomData;
use wgpu::util::DeviceExt;

/// Buffer usage flags
pub struct BufferUsage;

impl BufferUsage {
    pub const STORAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE;
    pub const UNIFORM: wgpu::BufferUsages = wgpu::BufferUsages::UNIFORM;
    pub const COPY_SRC: wgpu::BufferUsages = wgpu::BufferUsages::COPY_SRC;
    pub const COPY_DST: wgpu::BufferUsages = wgpu::BufferUsages::COPY_DST;
}

/// GPU buffer wrapper
#[derive(Debug)]
pub struct GpuBuffer {
    buffer: wgpu::Buffer,
    size: usize,
    usage: wgpu::BufferUsages,
}

impl GpuBuffer {
    /// Create empty buffer
    pub fn create(
        device: &wgpu::Device,
        size: usize,
        usage: wgpu::BufferUsages,
    ) -> KwaversResult<Self> {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kwavers_buffer"),
            size: size as u64,
            usage,
            mapped_at_creation: false,
        });

        Ok(Self {
            buffer,
            size,
            usage,
        })
    }

    /// Create buffer with data
    pub fn create_with_data<T: bytemuck::Pod>(
        device: &wgpu::Device,
        data: &[T],
        usage: wgpu::BufferUsages,
    ) -> KwaversResult<Self> {
        let bytes = bytemuck::cast_slice(data);

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("kwavers_buffer"),
            contents: bytes,
            usage,
        });

        Ok(Self {
            buffer,
            size: bytes.len(),
            usage,
        })
    }

    /// Write data to buffer
    pub fn write<T: bytemuck::Pod>(&self, queue: &wgpu::Queue, data: &[T]) {
        let bytes = bytemuck::cast_slice(data);
        queue.write_buffer(&self.buffer, 0, bytes);
    }

    /// Read buffer to vector
    pub async fn read_to_vec<T: bytemuck::Pod>(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> KwaversResult<Vec<T>> {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_buffer"),
            size: self.size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("buffer_read"),
        });

        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, self.size as u64);

        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        device.poll(wgpu::Maintain::Wait);

        receiver.recv().map_err(|e| {
            crate::KwaversError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to map buffer: {}", e),
            ))
        })??;

        let data = buffer_slice.get_mapped_range();
        let result = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging.unmap();

        Ok(result)
    }

    /// Get buffer reference
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Get buffer size
    pub fn size(&self) -> usize {
        self.size
    }
}
