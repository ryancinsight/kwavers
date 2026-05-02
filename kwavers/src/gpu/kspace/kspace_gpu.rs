//! `KSpaceGpu` — GPU-accelerated k-space solver using wgpu.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::Array3;

/// GPU-accelerated k-space solver
#[derive(Debug)]
pub struct KSpaceGpu {
    fft_pipeline: wgpu::ComputePipeline,
    propagate_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    _spectrum_buffer: wgpu::Buffer,
    kspace_buffer: wgpu::Buffer,
    workgroup_size: [u32; 3],
}

impl KSpaceGpu {
    /// Create a new k-space GPU solver
    pub fn new(device: &wgpu::Device, grid: &Grid) -> KwaversResult<Self> {
        let fft_shader = include_str!("../shaders/fft.wgsl");
        let propagate_shader = include_str!("../shaders/kspace_propagate.wgsl");

        let fft_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FFT Shader"),
            source: wgpu::ShaderSource::Wgsl(fft_shader.into()),
        });

        let propagate_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("K-Space Propagate Shader"),
            source: wgpu::ShaderSource::Wgsl(propagate_shader.into()),
        });

        let grid_size = (grid.nx * grid.ny * grid.nz) as u64;
        let complex_size = grid_size * 2 * std::mem::size_of::<f32>() as u64;

        let spectrum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spectrum Buffer"),
            size: complex_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let kspace_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("K-Space Buffer"),
            size: grid_size * std::mem::size_of::<f32>() as u64 * 3,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("K-Space Bind Group Layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("K-Space Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: spectrum_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: kspace_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("K-Space Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..20,
            }],
        });

        let fft_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FFT Pipeline"),
            layout: Some(&pipeline_layout),
            module: &fft_module,
            entry_point: Some("fft_forward"),
            compilation_options: Default::default(),
            cache: None,
        });

        let propagate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("K-Space Propagate Pipeline"),
            layout: Some(&pipeline_layout),
            module: &propagate_module,
            entry_point: Some("propagate"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            fft_pipeline,
            propagate_pipeline,
            bind_group,
            _spectrum_buffer: spectrum_buffer,
            kspace_buffer,
            workgroup_size: [8, 8, 8],
        })
    }

    /// Upload k-space vectors
    pub fn upload_kspace(
        &self,
        queue: &wgpu::Queue,
        kx: &Array3<f64>,
        ky: &Array3<f64>,
        kz: &Array3<f64>,
    ) {
        let mut k_data = Vec::new();
        for ((kx_val, ky_val), kz_val) in kx.iter().zip(ky.iter()).zip(kz.iter()) {
            k_data.push(*kx_val as f32);
            k_data.push(*ky_val as f32);
            k_data.push(*kz_val as f32);
        }
        queue.write_buffer(&self.kspace_buffer, 0, bytemuck::cast_slice(&k_data));
    }

    /// Perform k-space propagation step
    pub fn propagate(&self, encoder: &mut wgpu::CommandEncoder, grid: &Grid, dt: f32, c0: f32) {
        let push_constants = [
            grid.nx as u32,
            grid.ny as u32,
            grid.nz as u32,
            dt.to_bits(),
            c0.to_bits(),
        ];

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FFT Forward Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fft_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_push_constants(0, bytemuck::cast_slice(&push_constants));
            pass.dispatch_workgroups(
                (grid.nx as u32).div_ceil(self.workgroup_size[0]),
                (grid.ny as u32).div_ceil(self.workgroup_size[1]),
                (grid.nz as u32).div_ceil(self.workgroup_size[2]),
            );
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("K-Space Propagate Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.propagate_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_push_constants(0, bytemuck::cast_slice(&push_constants));
            pass.dispatch_workgroups(
                (grid.nx as u32).div_ceil(self.workgroup_size[0]),
                (grid.ny as u32).div_ceil(self.workgroup_size[1]),
                (grid.nz as u32).div_ceil(self.workgroup_size[2]),
            );
        }
    }
}
