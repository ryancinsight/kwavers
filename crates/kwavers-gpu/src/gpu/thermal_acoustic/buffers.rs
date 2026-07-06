//! Provider-owned thermal-acoustic GPU buffer allocation and field I/O.
//!
//! SRP: changes when the buffer set or I/O protocol changes.

use super::config::GpuThermalAcousticConfig;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3 as LetoArray3;
use std::mem;

/// Provider contract for thermal-acoustic buffer sets.
pub trait ThermalAcousticBufferProvider: std::fmt::Debug {
    /// Scalar stored by this provider's field buffers.
    type Scalar;

    /// Return the 3-D simulation dimensions.
    fn dims(&self) -> (usize, usize, usize);

    /// Return the scalar grid size.
    fn grid_size(&self) -> u64;
}

/// Provider-generic GPU buffers for thermal-acoustic coupling.
#[derive(Debug)]
pub struct GpuThermalAcousticBuffers<P = WgpuThermalAcousticBuffers>
where
    P: ThermalAcousticBufferProvider,
{
    provider: P,
}

impl GpuThermalAcousticBuffers<WgpuThermalAcousticBuffers> {
    /// Allocate WGPU thermal-acoustic buffers.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &GpuThermalAcousticConfig,
    ) -> KwaversResult<Self> {
        WgpuThermalAcousticBuffers::new(device, queue, config).map(Self::from_provider)
    }

    /// Upload initial conditions to WGPU buffers.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if field dimensions or layout are invalid.
    /// - Propagates any [`KwaversError`] returned by called functions.
    pub fn upload_fields(
        &self,
        queue: &wgpu::Queue,
        pressure: &LetoArray3<f32>,
        velocity_x: &LetoArray3<f32>,
        velocity_y: &LetoArray3<f32>,
        velocity_z: &LetoArray3<f32>,
        temperature: &LetoArray3<f32>,
    ) -> KwaversResult<()> {
        self.provider.upload_fields(
            queue,
            pressure,
            velocity_x,
            velocity_y,
            velocity_z,
            temperature,
        )
    }

    /// Download the current pressure, velocity_x, and temperature fields from WGPU buffers.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    pub async fn download_fields(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> KwaversResult<(LetoArray3<f32>, LetoArray3<f32>, LetoArray3<f32>)> {
        self.provider.download_fields(device, queue).await
    }
}

impl<P> GpuThermalAcousticBuffers<P>
where
    P: ThermalAcousticBufferProvider,
{
    /// Build a provider-generic buffer wrapper.
    #[must_use]
    pub const fn from_provider(provider: P) -> Self {
        Self { provider }
    }

    /// Borrow the concrete buffer provider.
    #[must_use]
    pub const fn provider(&self) -> &P {
        &self.provider
    }

    /// Return the 3-D simulation dimensions.
    #[must_use]
    pub fn dims(&self) -> (usize, usize, usize) {
        self.provider.dims()
    }

    /// Return the scalar grid size.
    #[must_use]
    pub fn grid_size(&self) -> u64 {
        self.provider.grid_size()
    }
}

/// WGPU buffers for thermal-acoustic coupling.
#[derive(Debug)]
pub struct WgpuThermalAcousticBuffers {
    pub dims: (usize, usize, usize),
    pub(super) pressure_curr: wgpu::Buffer,
    pub(super) pressure_prev: wgpu::Buffer,
    pub(super) velocity_x_curr: wgpu::Buffer,
    pub(super) velocity_y_curr: wgpu::Buffer,
    pub(super) velocity_z_curr: wgpu::Buffer,
    pub(super) velocity_x_prev: wgpu::Buffer,
    pub(super) velocity_y_prev: wgpu::Buffer,
    pub(super) velocity_z_prev: wgpu::Buffer,
    pub(super) temperature_curr: wgpu::Buffer,
    pub(super) temperature_prev: wgpu::Buffer,
    #[allow(non_snake_case)]
    pub(super) q_ac: wgpu::Buffer,
    pub(super) config_buffer: wgpu::Buffer,
    pub grid_size: u64,
}

impl ThermalAcousticBufferProvider for WgpuThermalAcousticBuffers {
    type Scalar = f32;

    fn dims(&self) -> (usize, usize, usize) {
        self.dims
    }

    fn grid_size(&self) -> u64 {
        self.grid_size
    }
}

impl WgpuThermalAcousticBuffers {
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
        pressure: &LetoArray3<f32>,
        velocity_x: &LetoArray3<f32>,
        velocity_y: &LetoArray3<f32>,
        velocity_z: &LetoArray3<f32>,
        temperature: &LetoArray3<f32>,
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

        let expected = [self.dims.0, self.dims.1, self.dims.2];
        if pressure.shape() != expected
            || velocity_x.shape() != expected
            || velocity_y.shape() != expected
            || velocity_z.shape() != expected
            || temperature.shape() != expected
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
    ) -> KwaversResult<(LetoArray3<f32>, LetoArray3<f32>, LetoArray3<f32>)> {
        async fn read_to_array3(
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            source: &wgpu::Buffer,
            staging: &wgpu::Buffer,
            dims: [usize; 3],
            buffer_size: u64,
        ) -> KwaversResult<LetoArray3<f32>> {
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
            result
                .map_err(|e| crate::gpu::map_buffer_async_error("thermal acoustic readback", e))?;

            let data = slice.get_mapped_range();
            let float_data: &[f32] = bytemuck::cast_slice(&data);
            let output = LetoArray3::from_shape_vec(dims, float_data.to_vec())
                .map_err(|e| KwaversError::GpuError(format!("Array creation error: {}", e)))?;
            drop(data);
            staging.unmap();
            Ok(output)
        }

        let buffer_size = self.grid_size * mem::size_of::<f32>() as u64;
        let shape = [self.dims.0, self.dims.1, self.dims.2];
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
            shape,
            buffer_size,
        )
        .await?;
        let velocity_x = read_to_array3(
            device,
            queue,
            &self.velocity_x_curr,
            &staging,
            shape,
            buffer_size,
        )
        .await?;
        let temperature = read_to_array3(
            device,
            queue,
            &self.temperature_curr,
            &staging,
            shape,
            buffer_size,
        )
        .await?;

        Ok((pressure, velocity_x, temperature))
    }
}
