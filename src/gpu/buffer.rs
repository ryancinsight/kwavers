//! GPU buffer management
//!
//! This module provides safe, high-level abstractions for GPU buffer operations including:
//! - Buffer creation with automatic memory management
//! - Asynchronous data transfer between CPU and GPU
//! - Type-safe buffer operations using bytemuck traits
//! - Staging buffer patterns for GPU-to-CPU reads
//!
//! # Examples
//!
//! ```no_run
//! # use kwavers::gpu::buffer::{GpuBuffer, BufferUsage};
//! # async fn example(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<(), Box<dyn std::error::Error>> {
//! // Create a buffer with initial data
//! let data = vec![1.0f32, 2.0, 3.0, 4.0];
//! let buffer = GpuBuffer::create_with_data(
//!     device,
//!     &data,
//!     BufferUsage::STORAGE | BufferUsage::COPY_SRC
//! )?;
//!
//! // Read data back from GPU
//! let result: Vec<f32> = buffer.read_to_vec(device, queue).await?;
//! assert_eq!(result, data);
//! # Ok(())
//! # }
//! ```

use crate::domain::core::error::{KwaversError, KwaversResult};
use wgpu::util::DeviceExt;

/// Buffer usage flags for GPU buffers
///
/// These constants define how a buffer will be used, which affects memory allocation
/// and allowed operations. Multiple flags can be combined using the bitwise OR operator.
///
/// # Examples
///
/// ```
/// # use kwavers::gpu::buffer::BufferUsage;
/// // Storage buffer that can be copied from
/// let usage = BufferUsage::STORAGE | BufferUsage::COPY_SRC;
/// ```
#[derive(Debug)]
pub struct BufferUsage;

impl BufferUsage {
    /// Storage buffer usage - can be bound as storage buffer in shaders
    pub const STORAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE;

    /// Uniform buffer usage - can be bound as uniform buffer in shaders
    pub const UNIFORM: wgpu::BufferUsages = wgpu::BufferUsages::UNIFORM;

    /// Copy source - can be used as source in copy operations
    pub const COPY_SRC: wgpu::BufferUsages = wgpu::BufferUsages::COPY_SRC;

    /// Copy destination - can be used as destination in copy operations
    pub const COPY_DST: wgpu::BufferUsages = wgpu::BufferUsages::COPY_DST;
}

/// GPU buffer wrapper providing safe, high-level buffer operations
///
/// `GpuBuffer` manages GPU memory allocation and provides type-safe operations
/// for transferring data between CPU and GPU. It handles the complexity of
/// staging buffers and async operations required for GPU-to-CPU data transfer.
///
/// # Buffer Usage Patterns
///
/// - **Storage Buffer**: Use `BufferUsage::STORAGE` for compute shader read/write
/// - **Uniform Buffer**: Use `BufferUsage::UNIFORM` for shader constants
/// - **GPU-to-CPU**: Requires `COPY_SRC` on main buffer, staging buffer internally managed
/// - **CPU-to-GPU**: Requires `COPY_DST` on main buffer
///
/// # Examples
///
/// ```no_run
/// # use kwavers::gpu::buffer::{GpuBuffer, BufferUsage};
/// # async fn example(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<(), Box<dyn std::error::Error>> {
/// // Create empty buffer for compute output
/// let buffer = GpuBuffer::create(
///     device,
///     1024,  // size in bytes
///     BufferUsage::STORAGE | BufferUsage::COPY_SRC
/// )?;
///
/// // Write data to buffer
/// let data = vec![1.0f32; 256];
/// buffer.write(queue, &data);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct GpuBuffer {
    buffer: wgpu::Buffer,
    size: usize,
    #[allow(dead_code)]
    usage: wgpu::BufferUsages,
}

impl GpuBuffer {
    /// Create an empty GPU buffer
    ///
    /// Allocates GPU memory of the specified size with given usage flags.
    /// The buffer contents are uninitialized.
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device for buffer allocation
    /// * `size` - Size of buffer in bytes
    /// * `usage` - Buffer usage flags (can combine multiple flags with `|`)
    ///
    /// # Returns
    ///
    /// Returns `Ok(GpuBuffer)` on success, or an error if allocation fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kwavers::gpu::buffer::{GpuBuffer, BufferUsage};
    /// # fn example(device: &wgpu::Device) -> Result<(), Box<dyn std::error::Error>> {
    /// let buffer = GpuBuffer::create(
    ///     device,
    ///     4096,  // 4KB buffer
    ///     BufferUsage::STORAGE | BufferUsage::COPY_DST
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
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

    /// Create a GPU buffer initialized with data
    ///
    /// Allocates GPU memory and immediately copies the provided data.
    /// The data type must implement `bytemuck::Pod` for safe byte-level copying.
    ///
    /// # Type Parameters
    ///
    /// * `T` - Data type implementing `bytemuck::Pod` (e.g., f32, u32, [f32; 4])
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device for buffer allocation
    /// * `data` - Slice of data to copy to GPU
    /// * `usage` - Buffer usage flags
    ///
    /// # Returns
    ///
    /// Returns `Ok(GpuBuffer)` with data copied, or an error if allocation fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kwavers::gpu::buffer::{GpuBuffer, BufferUsage};
    /// # fn example(device: &wgpu::Device) -> Result<(), Box<dyn std::error::Error>> {
    /// let data = vec![1.0f32, 2.0, 3.0, 4.0];
    /// let buffer = GpuBuffer::create_with_data(
    ///     device,
    ///     &data,
    ///     BufferUsage::STORAGE | BufferUsage::COPY_SRC
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
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

    /// Write data to GPU buffer
    ///
    /// Synchronously writes data from CPU to GPU using the queue.
    /// The buffer must have `COPY_DST` usage flag.
    ///
    /// # Type Parameters
    ///
    /// * `T` - Data type implementing `bytemuck::Pod`
    ///
    /// # Arguments
    ///
    /// * `queue` - GPU queue for write operation
    /// * `data` - Slice of data to write
    ///
    /// # Panics
    ///
    /// Panics if data size exceeds buffer size.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kwavers::gpu::buffer::{GpuBuffer, BufferUsage};
    /// # fn example(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &GpuBuffer) {
    /// let new_data = vec![5.0f32, 6.0, 7.0, 8.0];
    /// buffer.write(queue, &new_data);
    /// # }
    /// ```
    pub fn write<T: bytemuck::Pod>(&self, queue: &wgpu::Queue, data: &[T]) {
        let bytes = bytemuck::cast_slice(data);
        queue.write_buffer(&self.buffer, 0, bytes);
    }

    /// Read buffer data from GPU to CPU
    ///
    /// Asynchronously reads buffer contents using a staging buffer pattern.
    /// The buffer must have `COPY_SRC` usage flag. This operation:
    ///
    /// 1. Creates a staging buffer with `MAP_READ` + `COPY_DST` usage
    /// 2. Copies GPU buffer to staging buffer
    /// 3. Maps staging buffer to CPU memory
    /// 4. Returns data as Vec<T>
    ///
    /// # Type Parameters
    ///
    /// * `T` - Data type implementing `bytemuck::Pod`
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device for staging buffer creation
    /// * `queue` - GPU queue for copy operation
    ///
    /// # Returns
    ///
    /// Returns `Ok(Vec<T>)` with buffer contents, or an error if read fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use kwavers::gpu::buffer::{GpuBuffer, BufferUsage};
    /// # async fn example(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &GpuBuffer) -> Result<(), Box<dyn std::error::Error>> {
    /// let data: Vec<f32> = buffer.read_to_vec(device, queue).await?;
    /// println!("Read {} elements from GPU", data.len());
    /// # Ok(())
    /// # }
    /// ```
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
            KwaversError::Io(std::io::Error::other(format!("Failed to map buffer: {e}")))
        })??;

        let data = buffer_slice.get_mapped_range();
        let result = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging.unmap();

        Ok(result)
    }

    /// Get reference to underlying wgpu buffer
    ///
    /// Provides access to the raw `wgpu::Buffer` for advanced operations.
    ///
    /// # Returns
    ///
    /// Reference to the underlying wgpu buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Get buffer size in bytes
    ///
    /// # Returns
    ///
    /// Size of the buffer in bytes.
    pub fn size(&self) -> usize {
        self.size
    }
}
