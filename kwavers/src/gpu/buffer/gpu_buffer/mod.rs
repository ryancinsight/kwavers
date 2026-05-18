//! `GpuBuffer`: GPU buffer allocation, write, and accessor methods.

mod readback;

use once_cell::sync::OnceCell;
use wgpu::util::DeviceExt;

use crate::core::error::KwaversResult;

/// GPU buffer wrapper providing safe, high-level buffer operations
///
/// `GpuBuffer` manages GPU memory allocation and provides type-safe operations
/// for transferring data between CPU and GPU. It handles the complexity of
/// staging buffers and async operations required for GPU-to-CPU data transfer.
///
/// This is the canonical GPU buffer type for the kwavers codebase.
/// [`GpuBufferManager`](crate::gpu::GpuBufferManager) uses this type for named-buffer pools.
///
/// # Buffer Usage Patterns
///
/// - **Storage Buffer**: Use `BufferUsage::STORAGE` for compute shader read/write
/// - **Uniform Buffer**: Use `BufferUsage::UNIFORM` for shader constants
/// - **GPU-to-CPU**: Uses direct `MAP_READ` when available, otherwise `COPY_SRC`
///   on the main buffer with a cached staging buffer
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
    pub(super) buffer: wgpu::Buffer,
    pub(super) size: usize,
    /// Usage flags queried by `read_to_vec` to select MAP_READ vs COPY_SRC readback path.
    pub(super) usage: wgpu::BufferUsages,
    /// Lazily initialized readback staging buffer reused across `read_to_vec`
    /// calls. This avoids reallocating a fresh staging buffer for every read.
    pub(super) readback_staging: OnceCell<wgpu::Buffer>,
}

impl GpuBuffer {
    /// Create an empty, labeled GPU buffer (infallible).
    ///
    /// Primary constructor used by [`GpuBufferManager`](crate::gpu::GpuBufferManager) and
    /// any call site that requires a diagnostic label. `wgpu::Device::create_buffer`
    /// panics on OOM rather than returning an error, so this constructor is infallible.
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device for buffer allocation
    /// * `label`  - Debug label attached to the wgpu buffer
    /// * `size`   - Size of buffer in bytes
    /// * `usage`  - Buffer usage flags (combine with `|`)
    pub fn new(device: &wgpu::Device, label: &str, size: usize, usage: wgpu::BufferUsages) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size as u64,
            usage,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            size,
            usage,
            readback_staging: OnceCell::new(),
        }
    }

    /// Create a labeled GPU buffer initialized with `data` (infallible).
    ///
    /// Copies `data` to device memory at construction time.
    /// The data type must implement `bytemuck::Pod` for safe byte-level copying.
    ///
    /// # Type Parameters
    ///
    /// * `T` - Element type implementing `bytemuck::Pod`
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device for buffer allocation
    /// * `label`  - Debug label attached to the wgpu buffer
    /// * `data`   - Slice of data to copy to GPU
    /// * `usage`  - Buffer usage flags
    pub fn new_with_data<T: bytemuck::Pod>(
        device: &wgpu::Device,
        label: &str,
        data: &[T],
        usage: wgpu::BufferUsages,
    ) -> Self {
        let bytes = bytemuck::cast_slice(data);
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytes,
            usage,
        });
        Self {
            buffer,
            size: bytes.len(),
            usage,
            readback_staging: OnceCell::new(),
        }
    }

    /// Create an empty GPU buffer (unlabeled convenience form).
    ///
    /// Allocates GPU memory of the specified size with given usage flags.
    /// The buffer contents are uninitialized. Prefer [`GpuBuffer::new`] when
    /// a diagnostic label is available.
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn create(
        device: &wgpu::Device,
        size: usize,
        usage: wgpu::BufferUsages,
    ) -> KwaversResult<Self> {
        Ok(Self::new(device, "kwavers_buffer", size, usage))
    }

    /// Create a GPU buffer initialized with data (unlabeled convenience form).
    ///
    /// Allocates GPU memory and immediately copies the provided data.
    /// The data type must implement `bytemuck::Pod` for safe byte-level copying.
    /// Prefer [`GpuBuffer::new_with_data`] when a diagnostic label is available.
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn create_with_data<T: bytemuck::Pod>(
        device: &wgpu::Device,
        data: &[T],
        usage: wgpu::BufferUsages,
    ) -> KwaversResult<Self> {
        Ok(Self::new_with_data(device, "kwavers_buffer", data, usage))
    }

    /// Write data to GPU buffer.
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
    pub fn write<T: bytemuck::Pod>(&self, queue: &wgpu::Queue, data: &[T]) {
        let bytes = bytemuck::cast_slice(data);
        queue.write_buffer(&self.buffer, 0, bytes);
    }

    /// Get reference to underlying wgpu buffer.
    ///
    /// Provides access to the raw `wgpu::Buffer` for advanced operations.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Get buffer size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }
}
