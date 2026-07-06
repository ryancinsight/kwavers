//! PSTD buffer allocation provider contracts.

use wgpu::util::DeviceExt;

/// Provider contract for PSTD solver buffer allocation.
pub trait PstdBufferProvider {
    /// Provider-owned buffer type.
    type Buffer;

    /// Create a read-only storage buffer initialized from host data.
    fn read_only_storage<T>(&self, data: &[T], label: &'static str) -> Self::Buffer
    where
        T: bytemuck::Pod;

    /// Create a static read-only storage buffer initialized from host data.
    fn static_storage<T>(&self, data: &[T], label: &'static str) -> Self::Buffer
    where
        T: bytemuck::Pod;

    /// Create a storage buffer used for host-to-device uploads.
    fn upload_storage<T>(&self, data: &[T], label: &'static str) -> Self::Buffer
    where
        T: bytemuck::Pod;

    /// Create a read/write storage buffer.
    fn read_write_storage<T>(&self, len: usize, label: &'static str) -> Self::Buffer
    where
        T: bytemuck::Pod;

    /// Create a host-readable staging buffer.
    fn map_read_buffer<T>(&self, len: usize, label: &'static str) -> Self::Buffer
    where
        T: bytemuck::Pod;
}

/// WGPU PSTD buffer allocation provider.
pub struct WgpuPstdBufferFactory<'a> {
    device: &'a wgpu::Device,
}

impl<'a> WgpuPstdBufferFactory<'a> {
    /// Create a WGPU PSTD buffer factory.
    #[must_use]
    pub const fn new(device: &'a wgpu::Device) -> Self {
        Self { device }
    }
}

impl PstdBufferProvider for WgpuPstdBufferFactory<'_> {
    type Buffer = wgpu::Buffer;

    fn read_only_storage<T>(&self, data: &[T], label: &'static str) -> Self::Buffer
    where
        T: bytemuck::Pod,
    {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            })
    }

    fn static_storage<T>(&self, data: &[T], label: &'static str) -> Self::Buffer
    where
        T: bytemuck::Pod,
    {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
    }

    fn upload_storage<T>(&self, data: &[T], label: &'static str) -> Self::Buffer
    where
        T: bytemuck::Pod,
    {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
    }

    fn read_write_storage<T>(&self, len: usize, label: &'static str) -> Self::Buffer
    where
        T: bytemuck::Pod,
    {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (len * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn map_read_buffer<T>(&self, len: usize, label: &'static str) -> Self::Buffer
    where
        T: bytemuck::Pod,
    {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (len * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
}
