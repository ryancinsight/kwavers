//! Provider-trait FDTD acoustic solver boundary.
//!
//! The current concrete implementation is WGPU/WGSL because the real FDTD
//! kernels in this crate are WGSL. CUDA and future providers implement the
//! [`FdtdGpuProvider`] trait only after they own real kernels for this contract.
//!
//! `WgpuFdtd` uses `fdtd.wgsl` which provides two entry points dispatched in sequence:
//! 1. `velocity_update` — reads p, writes v_new (bindings 0, 1, 2)
//! 2. `pressure_update` — reads v_new, writes p_new (bindings 0, 1, 2)
//!
//! The two-pass split eliminates the cross-workgroup memory-ordering hazard
//! that arises when pressure and velocity are updated in a single kernel while
//! both buffers are bound as `read_write`.
//!
//! For second-order-accurate leapfrog FDTD, use
//! [`super::compute::WgpuFdtdPressureDispatcher`] which dispatches
//! `fdtd_pressure.wgsl` (three-buffer leapfrog, correct staggering).

use std::future::Future;

use crate::{
    backend::{
        init::GpuProviderContext,
        provider::{GpuKernelProvider, GpuProviderBackend},
    },
    gpu::GpuDeviceProvider,
};
use hephaestus_core::{DeviceFeature, DeviceLimits};
use hephaestus_wgpu::WgpuDevice;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use kwavers_solver::backend::traits::BackendCapabilities;
use leto::Array3 as LetoArray3;

/// Provider execution seam for FDTD pressure transfer and stepping.
///
/// This trait is generic over the Hephaestus provider through its associated
/// device type. WGPU implements it with real WGSL kernels;
/// CUDA must add a separate real implementation before satisfying this trait.
pub trait FdtdGpuProvider: GpuKernelProvider {
    /// Build the provider-specific FDTD solver for `grid`.
    ///
    /// # Errors
    ///
    /// Returns a GPU error when provider acquisition or kernel construction
    /// fails.
    fn new(grid: &Grid) -> KwaversResult<Self>
    where
        Self: Sized;

    /// Upload a provider-native pressure field to GPU memory.
    ///
    /// # Errors
    ///
    /// Returns an error when the pressure layout cannot be represented by the
    /// provider kernel contract.
    fn upload_pressure(&self, pressure: &LetoArray3<Self::Scalar>) -> KwaversResult<()>;

    /// Download the provider-native pressure field from GPU memory.
    ///
    /// # Errors
    ///
    /// Propagates provider transfer or readback failures.
    fn download_pressure<'a>(
        &'a self,
        grid: &'a Grid,
    ) -> impl Future<Output = KwaversResult<LetoArray3<Self::Scalar>>> + 'a;

    /// Download the provider-native pressure field without requiring callers
    /// to own an async runtime.
    ///
    /// # Errors
    ///
    /// Propagates provider transfer or readback failures.
    fn download_pressure_blocking(&self, grid: &Grid) -> KwaversResult<LetoArray3<Self::Scalar>> {
        pollster::block_on(self.download_pressure(grid))
    }

    /// Encode one FDTD step into the provider command stream.
    ///
    /// # Errors
    ///
    /// Returns a GPU error when the provider cannot encode the step.
    fn step(&self, grid: &Grid, dt: Self::Scalar) -> KwaversResult<()>;
}

/// WGPU FDTD solver: two-pass collocated Euler split.
///
/// Step sequence per time step:
///   1. Dispatch `velocity_update` pipeline → writes updated velocity.
///   2. Dispatch `pressure_update` pipeline → reads updated velocity, writes updated pressure.
///
/// Both pipelines share the same bind group (same buffers, same layout).
/// Separation at the command-encoder level provides the required synchronization
/// barrier between the two passes.
#[derive(Debug)]
pub struct WgpuFdtd {
    context: GpuProviderContext<WgpuDevice>,
    velocity_pipeline: wgpu::ComputePipeline,
    pressure_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    pressure_buffer: wgpu::Buffer,
    /// GPU resource lifetime anchor — bind group references this buffer on-device.
    _velocity_buffer: wgpu::Buffer,
    /// GPU resource lifetime anchor — bind group references this buffer on-device.
    _medium_buffer: wgpu::Buffer,
    workgroup_size: [u32; 3],
}

impl WgpuFdtd {
    /// Build GPU pipelines for the two-pass FDTD solver.
    ///
    /// Compiles `fdtd.wgsl` once; creates separate `ComputePipeline` objects for
    /// `velocity_update` and `pressure_update` entry points from the same module.
    ///
    /// # Errors
    /// Propagates [`KwaversError::GpuError`] on pipeline creation failure.
    pub fn new(grid: &Grid) -> KwaversResult<Self> {
        <Self as FdtdGpuProvider>::new(grid)
    }

    fn build(grid: &Grid) -> KwaversResult<Self> {
        let context = GpuProviderContext::<WgpuDevice>::with_features_and_limits(
            WgpuDevice::acquisition_preference(),
            &[DeviceFeature::ImmediateData],
            fdtd_required_limits(),
        )?;
        let device = context.device();
        let shader_source = include_str!("shaders/fdtd.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fdtd"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let grid_size = (grid.nx * grid.ny * grid.nz) as u64;
        let float_bytes = std::mem::size_of::<f32>() as u64;
        let buffer_size = grid_size * float_bytes;

        let pressure_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fdtd_pressure"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // vec3<f32> per voxel = 3 × f32, but wgpu uses vec3 with 16-byte stride in storage
        let velocity_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fdtd_velocity"),
            size: buffer_size * 4, // 4 f32 per vec3 (std430 vec3 = 12B, but aligned to 16)
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // medium[idx] = vec2<f32>(rho0, c0)
        let medium_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fdtd_medium"),
            size: buffer_size * 2,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Both entry points share identical bind group layout (same bindings 0–2).
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fdtd_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fdtd_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pressure_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: velocity_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: medium_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fdtd_layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 16, // GridParams: nx, ny, nz (u32), dt (f32)
        });

        let velocity_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fdtd_velocity_update"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("velocity_update"),
            compilation_options: Default::default(),
            cache: None,
        });

        let pressure_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fdtd_pressure_update"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("pressure_update"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            context,
            velocity_pipeline,
            pressure_pipeline,
            bind_group,
            pressure_buffer,
            _velocity_buffer: velocity_buffer,
            _medium_buffer: medium_buffer,
            workgroup_size: [8, 8, 4],
        })
    }

    /// Upload a provider-native pressure field to GPU.
    ///
    /// # Errors
    /// Returns `KwaversError::InvalidInput` when the Leto field is not stored
    /// as one dense row-major slice.
    pub fn upload_pressure(&self, pressure: &LetoArray3<f32>) -> KwaversResult<()> {
        <Self as FdtdGpuProvider>::upload_pressure(self, pressure)
    }

    /// Download a provider-native pressure field from GPU.
    ///
    /// # Errors
    /// Propagates [`KwaversError::GpuError`] on GPU read-back failure.
    pub async fn download_pressure(&self, grid: &Grid) -> KwaversResult<LetoArray3<f32>> {
        self.download_pressure_impl(grid).await
    }

    /// Download a provider-native pressure field from GPU without requiring
    /// callers to own an async runtime.
    ///
    /// # Errors
    /// Propagates [`KwaversError::GpuError`] on GPU read-back failure.
    pub fn download_pressure_blocking(&self, grid: &Grid) -> KwaversResult<LetoArray3<f32>> {
        <Self as FdtdGpuProvider>::download_pressure_blocking(self, grid)
    }

    async fn download_pressure_impl(&self, grid: &Grid) -> KwaversResult<LetoArray3<f32>> {
        let size = (grid.nx * grid.ny * grid.nz * std::mem::size_of::<f32>()) as u64;
        let device = self.context.device();
        let queue = self.context.queue();

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fdtd_staging"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fdtd_readback"),
        });
        encoder.copy_buffer_to_buffer(&self.pressure_buffer, 0, &staging, 0, size);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = flume::bounded(1);
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv_async()
            .await
            .map_err(|e| KwaversError::GpuError(format!("readback channel: {e}")))?
            .map_err(|e| KwaversError::GpuError(format!("readback map: {e}")))?;

        let data = slice
            .get_mapped_range()
            .map_err(|error| crate::gpu::map_buffer_range_error("FDTD pressure readback", error))?;
        let floats: &[f32] = bytemuck::cast_slice(&data);
        let result_values = floats.to_vec();
        drop(data);
        staging.unmap();
        LetoArray3::from_shape_vec([grid.nx, grid.ny, grid.nz], result_values).map_err(|e| {
            KwaversError::InvalidInput(format!("Invalid FDTD pressure readback shape: {e}"))
        })
    }

    /// Encode one FDTD time step: velocity_update → barrier → pressure_update.
    ///
    /// The command encoder imposes the required ordering barrier between the two
    /// compute passes, preventing the pressure_update kernel from reading velocity
    /// values before velocity_update has finished writing them.
    pub fn encode_step(&self, encoder: &mut wgpu::CommandEncoder, grid: &Grid, dt: f32) {
        let push = [grid.nx as u32, grid.ny as u32, grid.nz as u32, dt.to_bits()];
        let wg_x = (grid.nx as u32).div_ceil(self.workgroup_size[0]);
        let wg_y = (grid.ny as u32).div_ceil(self.workgroup_size[1]);
        let wg_z = (grid.nz as u32).div_ceil(self.workgroup_size[2]);

        // Pass 1: velocity_update — reads pressure, writes velocity
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fdtd_velocity"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.velocity_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_immediates(0, bytemuck::cast_slice(&push));
            pass.dispatch_workgroups(wg_x, wg_y, wg_z);
        } // compute pass ends → implicit pipeline barrier

        // Pass 2: pressure_update — reads updated velocity, writes pressure
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fdtd_pressure"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pressure_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_immediates(0, bytemuck::cast_slice(&push));
            pass.dispatch_workgroups(wg_x, wg_y, wg_z);
        }
    }
}

impl GpuProviderBackend for WgpuFdtd {
    type Device = WgpuDevice;

    fn hephaestus_device(&self) -> &Self::Device {
        self.context.hephaestus_device()
    }

    fn device_name(&self) -> &str {
        self.context.device_name()
    }

    fn synchronize(&self) -> KwaversResult<()> {
        self.context.synchronize()
    }
}

impl GpuKernelProvider for WgpuFdtd {
    type Scalar = f32;

    fn capabilities(&self) -> BackendCapabilities {
        let limits = self.context.hephaestus_device().device_limits();
        BackendCapabilities {
            supports_fft: false,
            supports_f64: false,
            supports_f32: true,
            supports_async: true,
            max_parallelism: limits.max_compute_invocations_per_workgroup as usize,
            supports_unified_memory: false,
        }
    }

    fn is_available(&self) -> bool {
        true
    }

    fn available_memory(&self) -> usize {
        match usize::try_from(
            self.context
                .hephaestus_device()
                .device_limits()
                .max_buffer_size,
        ) {
            Ok(bytes) => bytes,
            Err(_) => usize::MAX,
        }
    }

    fn estimate_peak_performance(&self) -> f64 {
        0.0
    }
}

impl FdtdGpuProvider for WgpuFdtd {
    fn new(grid: &Grid) -> KwaversResult<Self> {
        Self::build(grid)
    }

    fn upload_pressure(&self, pressure: &LetoArray3<Self::Scalar>) -> KwaversResult<()> {
        let data = pressure.as_slice().ok_or_else(|| {
            KwaversError::InvalidInput(
                "FDTD pressure field must be dense row-major Leto Array3".to_string(),
            )
        })?;
        self.context
            .queue()
            .write_buffer(&self.pressure_buffer, 0, bytemuck::cast_slice(data));
        Ok(())
    }

    fn download_pressure<'a>(
        &'a self,
        grid: &'a Grid,
    ) -> impl Future<Output = KwaversResult<LetoArray3<Self::Scalar>>> + 'a {
        self.download_pressure_impl(grid)
    }

    fn step(&self, grid: &Grid, dt: Self::Scalar) -> KwaversResult<()> {
        let device = self.context.device();
        let queue = self.context.queue();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fdtd_step"),
        });
        self.encode_step(&mut encoder, grid, dt);
        queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }
}

fn fdtd_required_limits() -> DeviceLimits {
    DeviceLimits {
        max_immediate_size: 128,
        ..WgpuDevice::required_limits()
    }
}
