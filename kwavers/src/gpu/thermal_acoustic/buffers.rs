//! `GpuThermalAcousticBuffers` — GPU buffer allocation and field I/O.
//!
//! SRP: changes when the buffer set or I/O protocol changes.

use super::config::GpuThermalAcousticConfig;
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;
use std::mem;

/// GPU buffers for thermal-acoustic coupling
#[derive(Debug)]
pub struct GpuThermalAcousticBuffers {
    pub dims: (usize, usize, usize),
    pub pressure_curr: wgpu::Buffer,
    pub pressure_prev: wgpu::Buffer,
    pub velocity_x_curr: wgpu::Buffer,
    pub velocity_y_curr: wgpu::Buffer,
    pub velocity_z_curr: wgpu::Buffer,
    pub velocity_x_prev: wgpu::Buffer,
    pub velocity_y_prev: wgpu::Buffer,
    pub velocity_z_prev: wgpu::Buffer,
    pub temperature_curr: wgpu::Buffer,
    pub temperature_prev: wgpu::Buffer,
    #[allow(non_snake_case)]
    pub q_ac: wgpu::Buffer,
    pub config_buffer: wgpu::Buffer,
    pub grid_size: u64,
}

impl GpuThermalAcousticBuffers {
    /// New.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &GpuThermalAcousticConfig,
    ) -> KwaversResult<Self> {
        config.validate()?;

        let grid_size = (config.nx as u64) * (config.ny as u64) * (config.nz as u64);
        let buffer_size = grid_size * mem::size_of::<f32>() as u64;
        let dims = (config.nx as usize, config.ny as usize, config.nz as usize);

        let create_storage = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: buffer_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        let pressure_curr = create_storage("Pressure Current");
        let pressure_prev = create_storage("Pressure Previous");
        let velocity_x_curr = create_storage("Velocity X Current");
        let velocity_y_curr = create_storage("Velocity Y Current");
        let velocity_z_curr = create_storage("Velocity Z Current");
        let velocity_x_prev = create_storage("Velocity X Previous");
        let velocity_y_prev = create_storage("Velocity Y Previous");
        let velocity_z_prev = create_storage("Velocity Z Previous");
        let temperature_curr = create_storage("Temperature Current");
        let temperature_prev = create_storage("Temperature Previous");
        let q_ac = create_storage("Acoustic Heating");

        let config_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Thermal-Acoustic Config"),
            size: mem::size_of::<GpuThermalAcousticConfig>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&config_buffer, 0, bytemuck::bytes_of(config));

        Ok(Self {
            dims,
            pressure_curr,
            pressure_prev,
            velocity_x_curr,
            velocity_y_curr,
            velocity_z_curr,
            velocity_x_prev,
            velocity_y_prev,
            velocity_z_prev,
            temperature_curr,
            temperature_prev,
            q_ac,
            config_buffer,
            grid_size,
        })
    }

    /// Upload initial conditions to GPU
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn upload_fields(
        &self,
        queue: &wgpu::Queue,
        pressure: &Array3<f32>,
        velocity_x: &Array3<f32>,
        velocity_y: &Array3<f32>,
        velocity_z: &Array3<f32>,
        temperature: &Array3<f32>,
    ) -> KwaversResult<()> {
        let p_data = pressure.as_slice().ok_or_else(|| {
            KwaversError::InvalidInput("Pressure field must be contiguous".to_string())
        })?;
        let vx_data = velocity_x.as_slice().ok_or_else(|| {
            KwaversError::InvalidInput("Velocity X field must be contiguous".to_string())
        })?;
        let vy_data = velocity_y.as_slice().ok_or_else(|| {
            KwaversError::InvalidInput("Velocity Y field must be contiguous".to_string())
        })?;
        let vz_data = velocity_z.as_slice().ok_or_else(|| {
            KwaversError::InvalidInput("Velocity Z field must be contiguous".to_string())
        })?;
        let t_data = temperature.as_slice().ok_or_else(|| {
            KwaversError::InvalidInput("Temperature field must be contiguous".to_string())
        })?;

        let expected = self.dims;
        if pressure.dim() != expected
            || velocity_x.dim() != expected
            || velocity_y.dim() != expected
            || velocity_z.dim() != expected
            || temperature.dim() != expected
        {
            return Err(KwaversError::InvalidInput(
                "Field dimensions mismatch".to_string(),
            ));
        }

        queue.write_buffer(&self.pressure_curr, 0, bytemuck::cast_slice(p_data));
        queue.write_buffer(&self.velocity_x_curr, 0, bytemuck::cast_slice(vx_data));
        queue.write_buffer(&self.velocity_y_curr, 0, bytemuck::cast_slice(vy_data));
        queue.write_buffer(&self.velocity_z_curr, 0, bytemuck::cast_slice(vz_data));
        queue.write_buffer(&self.temperature_curr, 0, bytemuck::cast_slice(t_data));

        Ok(())
    }

    /// Download the current pressure, velocity_x, and temperature fields from GPU.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub async fn download_fields(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> KwaversResult<(Array3<f32>, Array3<f32>, Array3<f32>)> {
        async fn read_to_array3(
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            source: &wgpu::Buffer,
            staging: &wgpu::Buffer,
            dims: (usize, usize, usize),
            buffer_size: u64,
        ) -> KwaversResult<Array3<f32>> {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Download Encoder"),
            });
            encoder.copy_buffer_to_buffer(source, 0, staging, 0, buffer_size);
            queue.submit(std::iter::once(encoder.finish()));

            let slice = staging.slice(..buffer_size);
            let (tx, rx) = flume::bounded(1);
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            let _ = device.poll(wgpu::PollType::Wait);
            let result = rx
                .recv_async()
                .await
                .map_err(|e| KwaversError::GpuError(format!("Channel error: {}", e)))?;
            result?;

            let data = slice.get_mapped_range();
            let float_data: &[f32] = bytemuck::cast_slice(&data);
            let output = Array3::from_shape_vec(dims, float_data.to_vec())
                .map_err(|e| KwaversError::GpuError(format!("Array creation error: {}", e)))?;
            drop(data);
            staging.unmap();
            Ok(output)
        }

        let buffer_size = self.grid_size * mem::size_of::<f32>() as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let pressure = read_to_array3(
            device,
            queue,
            &self.pressure_curr,
            &staging,
            self.dims,
            buffer_size,
        )
        .await?;
        let velocity_x = read_to_array3(
            device,
            queue,
            &self.velocity_x_curr,
            &staging,
            self.dims,
            buffer_size,
        )
        .await?;
        let temperature = read_to_array3(
            device,
            queue,
            &self.temperature_curr,
            &staging,
            self.dims,
            buffer_size,
        )
        .await?;

        Ok((pressure, velocity_x, temperature))
    }
}
