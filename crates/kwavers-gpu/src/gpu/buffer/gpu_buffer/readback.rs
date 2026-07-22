//! `GpuBufferData::read_to_vec`: async GPU-to-CPU readback with lazy staging buffer.

use kwavers_core::error::{KwaversError, KwaversResult};

use crate::gpu::{CoreGpuContext, GpuDevice};

use super::GpuBufferData;

impl GpuBufferData {
    /// Read buffer data from GPU to CPU.
    ///
    /// Asynchronously reads buffer contents.
    /// If the buffer was created with `MAP_READ`, the buffer is mapped directly.
    /// Otherwise the buffer must have `COPY_SRC` usage and a staging buffer is
    /// allocated lazily and cached for reuse after the first read. This operation:
    ///
    /// 1. Maps the buffer directly when `MAP_READ` is available, or
    /// 2. Creates a staging buffer with `MAP_READ` + `COPY_DST` usage
    /// 3. Copies GPU buffer to staging buffer
    /// 4. Maps the staging buffer to CPU memory
    /// 5. Returns data as `Vec<T>`
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
    /// # Errors
    /// - Returns `KwaversError::System` if the precondition for a System-class constraint is violated.
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub async fn read_to_vec<T: bytemuck::Pod>(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> KwaversResult<Vec<T>> {
        if self.size == 0 {
            return Ok(Vec::new());
        }

        let elem_size = std::mem::size_of::<T>();
        if elem_size == 0 {
            return Err(KwaversError::System(
                kwavers_core::error::SystemError::InvalidOperation {
                    operation: "GPU buffer readback".to_string(),
                    reason: "Zero-sized element types are not supported".to_string(),
                },
            ));
        }

        if !self.size.is_multiple_of(elem_size) {
            return Err(KwaversError::System(
                kwavers_core::error::SystemError::InvalidOperation {
                    operation: "GPU buffer readback".to_string(),
                    reason: format!(
                        "Buffer byte size {} is not a multiple of element size {}",
                        self.size, elem_size
                    ),
                },
            ));
        }

        if self.usage.contains(wgpu::BufferUsages::MAP_READ) {
            let buffer_slice = self.buffer.slice(..);
            let (sender, receiver) = flume::bounded(1);
            // Flush any queued `write_buffer` staging copy before mapping.
            queue.submit([]);
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });

            let _ = device.poll(wgpu::PollType::wait_indefinitely());

            receiver
                .recv_async()
                .await
                .map_err(|e| {
                    KwaversError::Io(std::io::Error::other(format!("Failed to map buffer: {e}")))
                })?
                .map_err(|e| crate::gpu::map_buffer_async_error("primary buffer readback", e))?;

            let data = buffer_slice.get_mapped_range().map_err(|error| {
                crate::gpu::map_buffer_range_error("primary buffer readback", error)
            })?;
            let result = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            self.buffer.unmap();
            return Ok(result);
        }

        if !self.usage.contains(wgpu::BufferUsages::COPY_SRC) {
            return Err(KwaversError::System(
                kwavers_core::error::SystemError::InvalidOperation {
                    operation: "GPU buffer readback".to_string(),
                    reason: "Buffer not created with COPY_SRC usage".to_string(),
                },
            ));
        }

        let staging = self.readback_staging.get_or_init(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging_buffer"),
                size: self.size as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("buffer_read"),
        });

        encoder.copy_buffer_to_buffer(&self.buffer, 0, staging, 0, self.size as u64);

        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        receiver
            .recv_async()
            .await
            .map_err(|e| {
                KwaversError::Io(std::io::Error::other(format!("Failed to map buffer: {e}")))
            })?
            .map_err(|e| crate::gpu::map_buffer_async_error("staging buffer readback", e))?;

        let data = buffer_slice.get_mapped_range().map_err(|error| {
            crate::gpu::map_buffer_range_error("staging buffer readback", error)
        })?;
        let result = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging.unmap();

        Ok(result)
    }

    /// Read buffer data through a provider-owned context without requiring the
    /// caller to own an async runtime.
    ///
    /// # Errors
    /// - Returns `KwaversError::System` if the readback preconditions are
    ///   violated.
    /// - Propagates any `KwaversError` returned by GPU readback.
    pub fn read_to_vec_in_context<T: bytemuck::Pod>(
        &self,
        context: &CoreGpuContext,
    ) -> KwaversResult<Vec<T>> {
        pollster::block_on(self.read_to_vec(context.device(), context.queue()))
    }

    /// Read buffer data through a provider-owned device without requiring the
    /// caller to own an async runtime.
    ///
    /// # Errors
    /// - Returns `KwaversError::System` if the readback preconditions are
    ///   violated.
    /// - Propagates any `KwaversError` returned by GPU readback.
    pub fn read_to_vec_on_device<T: bytemuck::Pod>(
        &self,
        device: &GpuDevice,
    ) -> KwaversResult<Vec<T>> {
        pollster::block_on(self.read_to_vec(device.wgpu_device(), device.wgpu_queue()))
    }
}
