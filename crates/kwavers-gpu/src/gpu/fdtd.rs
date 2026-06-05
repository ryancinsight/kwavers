//! GPU-accelerated FDTD acoustic solver (collocated, first-order Euler split).
//!
//! Uses `fdtd.wgsl` which provides two entry points dispatched in sequence:
//! 1. `velocity_update` — reads p, writes v_new (bindings 0, 1, 2)
//! 2. `pressure_update` — reads v_new, writes p_new (bindings 0, 1, 2)
//!
//! The two-pass split eliminates the cross-workgroup memory-ordering hazard
//! that arises when pressure and velocity are updated in a single kernel while
//! both buffers are bound as `read_write`.
//!
//! For second-order-accurate leapfrog FDTD, use [`super::compute::FdtdGpuShaderDispatcher`]
//! which dispatches `fdtd_pressure.wgsl` (three-buffer leapfrog, correct staggering).

use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use ndarray::Array3;

/// GPU-accelerated FDTD solver: two-pass collocated Euler split.
///
/// Step sequence per time step:
///   1. Dispatch `velocity_update` pipeline → writes updated velocity.
///   2. Dispatch `pressure_update` pipeline → reads updated velocity, writes updated pressure.
///
/// Both pipelines share the same bind group (same buffers, same layout).
/// Separation at the command-encoder level provides the required synchronization
/// barrier between the two passes.
#[derive(Debug)]
pub struct FdtdGpu {
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

impl FdtdGpu {
    /// Build GPU pipelines for the two-pass FDTD solver.
    ///
    /// Compiles `fdtd.wgsl` once; creates separate `ComputePipeline` objects for
    /// `velocity_update` and `pressure_update` entry points from the same module.
    ///
    /// # Errors
    /// Propagates [`KwaversError::GpuError`] on pipeline creation failure.
    pub fn new(device: &wgpu::Device, grid: &Grid) -> KwaversResult<Self> {
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

        let push_range = wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..16, // GridParams: nx, ny, nz (u32), dt (f32) = 16 bytes
        };

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fdtd_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[push_range],
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
            velocity_pipeline,
            pressure_pipeline,
            bind_group,
            pressure_buffer,
            _velocity_buffer: velocity_buffer,
            _medium_buffer: medium_buffer,
            workgroup_size: [8, 8, 4],
        })
    }

    /// Upload pressure field to GPU.
    pub fn upload_pressure(&self, queue: &wgpu::Queue, pressure: &Array3<f64>) {
        let data: Vec<f32> = pressure.iter().map(|&x| x as f32).collect();
        queue.write_buffer(&self.pressure_buffer, 0, bytemuck::cast_slice(&data));
    }

    /// Download pressure field from GPU.
    ///
    /// # Errors
    /// Propagates [`KwaversError::GpuError`] on GPU read-back failure.
    pub async fn download_pressure(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let size = (grid.nx * grid.ny * grid.nz * std::mem::size_of::<f32>()) as u64;

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
        let _ = device.poll(wgpu::PollType::Wait);
        rx.recv_async()
            .await
            .map_err(|e| KwaversError::GpuError(format!("readback channel: {e}")))?
            .map_err(|e| KwaversError::GpuError(format!("readback map: {e}")))?;

        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        let plane = grid.ny * grid.nz;
        let result = Array3::from_shape_fn((grid.nx, grid.ny, grid.nz), |(i, j, k)| {
            floats[i * plane + j * grid.nz + k] as f64
        });
        drop(data);
        staging.unmap();
        Ok(result)
    }

    /// Encode one FDTD time step: velocity_update → barrier → pressure_update.
    ///
    /// The command encoder imposes the required ordering barrier between the two
    /// compute passes, preventing the pressure_update kernel from reading velocity
    /// values before velocity_update has finished writing them.
    pub fn step(&self, encoder: &mut wgpu::CommandEncoder, grid: &Grid, dt: f32) {
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
            pass.set_push_constants(0, bytemuck::cast_slice(&push));
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
            pass.set_push_constants(0, bytemuck::cast_slice(&push));
            pass.dispatch_workgroups(wg_x, wg_y, wg_z);
        }
    }
}
