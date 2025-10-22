//! GPU-accelerated FDTD solver

use crate::error::{KwaversError, KwaversResult};
use crate::grid::Grid;
use ndarray::Array3;

/// GPU-accelerated FDTD solver
/// NOTE: Some fields currently unused - part of future GPU pipeline implementation
#[allow(dead_code)]
#[derive(Debug)]
pub struct FdtdGpu {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    pressure_buffer: wgpu::Buffer,
    velocity_buffer: wgpu::Buffer,
    medium_buffer: wgpu::Buffer,
    workgroup_size: [u32; 3],
}

impl FdtdGpu {
    /// Create a new GPU FDTD solver
    pub fn new(device: &wgpu::Device, grid: &Grid) -> KwaversResult<Self> {
        let shader_source = include_str!("shaders/fdtd.wgsl");

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FDTD Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Calculate buffer sizes
        let grid_size = (grid.nx * grid.ny * grid.nz) as u64;
        let buffer_size = grid_size * std::mem::size_of::<f32>() as u64;

        // Create buffers
        let pressure_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pressure Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let velocity_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Velocity Buffer"),
            size: buffer_size * 3, // 3 components
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let medium_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Medium Buffer"),
            size: buffer_size * 2, // density and sound speed
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FDTD Bind Group Layout"),
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

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FDTD Bind Group"),
            layout: &bind_group_layout,
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

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("FDTD Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..16, // Grid dimensions and dt
            }],
        });

        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FDTD Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });

        let workgroup_size = [8, 8, 8];

        Ok(Self {
            pipeline,
            bind_group,
            pressure_buffer,
            velocity_buffer,
            medium_buffer,
            workgroup_size,
        })
    }

    /// Upload pressure field to GPU
    pub fn upload_pressure(&self, queue: &wgpu::Queue, pressure: &Array3<f64>) {
        let data: Vec<f32> = pressure.iter().map(|&x| x as f32).collect();
        queue.write_buffer(&self.pressure_buffer, 0, bytemuck::cast_slice(&data));
    }

    /// Download pressure field from GPU
    pub async fn download_pressure(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let size = (grid.nx * grid.ny * grid.nz * std::mem::size_of::<f32>()) as u64;

        // Create staging buffer
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Copy from GPU buffer to staging buffer
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Copy Encoder"),
        });

        encoder.copy_buffer_to_buffer(&self.pressure_buffer, 0, &staging_buffer, 0, size);

        // Submit commands
        queue.submit(std::iter::once(encoder.finish()));

        // Map and read buffer
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = flume::bounded(1);

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        device.poll(wgpu::Maintain::Wait);
        let result = rx
            .recv_async()
            .await
            .map_err(|e| KwaversError::GpuError(format!("Channel receive error: {}", e)))?;
        result?;

        let data = buffer_slice.get_mapped_range();
        let float_data: &[f32] = bytemuck::cast_slice(&data);

        let mut result = Array3::zeros((grid.nx, grid.ny, grid.nz));
        for (i, &val) in float_data.iter().enumerate() {
            let iz = i / (grid.nx * grid.ny);
            let iy = (i % (grid.nx * grid.ny)) / grid.nx;
            let ix = i % grid.nx;
            result[[ix, iy, iz]] = val as f64;
        }

        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }

    /// Run one FDTD time step
    pub fn step(&self, encoder: &mut wgpu::CommandEncoder, grid: &Grid, dt: f32) {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FDTD Compute Pass"),
            timestamp_writes: None,
        });

        // Set push constants (grid dimensions and dt)
        let push_constants = [grid.nx as u32, grid.ny as u32, grid.nz as u32, dt.to_bits()];

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.bind_group, &[]);
        compute_pass.set_push_constants(0, bytemuck::cast_slice(&push_constants));

        // Calculate workgroup counts
        let workgroups_x = (grid.nx as u32).div_ceil(self.workgroup_size[0]);
        let workgroups_y = (grid.ny as u32).div_ceil(self.workgroup_size[1]);
        let workgroups_z = (grid.nz as u32).div_ceil(self.workgroup_size[2]);

        compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
    }
}
