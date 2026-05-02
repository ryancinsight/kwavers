//! `GpuThermalAcousticSolver` — struct, construction, and time-stepping.
//!
//! SRP: changes when the pipeline layout, workgroup dispatch, or time-step
//! protocol changes.

use super::buffers::GpuThermalAcousticBuffers;
use super::config::GpuThermalAcousticConfig;
use super::shader::thermal_acoustic_wgsl;
use crate::core::error::KwaversResult;

/// GPU-accelerated thermal-acoustic coupling solver
#[derive(Debug)]
pub struct GpuThermalAcousticSolver {
    pub(super) config: GpuThermalAcousticConfig,
    pub(super) buffers: GpuThermalAcousticBuffers,
    pub(super) pipeline: wgpu::ComputePipeline,
    pub(super) bind_group: wgpu::BindGroup,
    pub(super) workgroup_size: [u32; 3],
}

impl GpuThermalAcousticSolver {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: GpuThermalAcousticConfig,
    ) -> KwaversResult<Self> {
        config.validate()?;
        let buffers = GpuThermalAcousticBuffers::new(device, queue, &config)?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Thermal-Acoustic Fused Kernel"),
            source: wgpu::ShaderSource::Wgsl(thermal_acoustic_wgsl().into()),
        });

        let bgl = build_bgl(device);
        let bind_group = build_bind_group(device, &bgl, &buffers);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Thermal-Acoustic Pipeline Layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Thermal-Acoustic Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            config,
            buffers,
            pipeline,
            bind_group,
            workgroup_size: [8, 8, 4],
        })
    }

    /// Execute one time step of the coupled simulation.
    pub fn step(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> KwaversResult<()> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Thermal-Acoustic Step Encoder"),
        });
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Thermal-Acoustic Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.bind_group, &[]);

        let workgroups_x = self.config.nx.div_ceil(self.workgroup_size[0]);
        let workgroups_y = self.config.ny.div_ceil(self.workgroup_size[1]);
        let workgroups_z = self.config.nz.div_ceil(self.workgroup_size[2]);
        compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        drop(compute_pass);
        queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    pub fn config(&self) -> GpuThermalAcousticConfig {
        self.config
    }
    pub fn buffers(&self) -> &GpuThermalAcousticBuffers {
        &self.buffers
    }
}

fn build_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let rw = |b: u32| wgpu::BindGroupLayoutEntry {
        binding: b,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let ro = |b: u32| wgpu::BindGroupLayoutEntry {
        binding: b,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let uniform = |b: u32| wgpu::BindGroupLayoutEntry {
        binding: b,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Thermal-Acoustic Bind Group Layout"),
        entries: &[
            rw(0),
            ro(1), // pressure curr/prev
            rw(2),
            ro(3), // velocity_x curr/prev
            rw(4),
            ro(5), // velocity_y curr/prev
            rw(6),
            ro(7), // velocity_z curr/prev
            rw(8),
            ro(9),       // temperature curr/prev
            rw(10),      // q_ac
            uniform(11), // config
        ],
    })
}

fn build_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buf: &GpuThermalAcousticBuffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Thermal-Acoustic Bind Group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.pressure_curr.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf.pressure_prev.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf.velocity_x_curr.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buf.velocity_x_prev.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: buf.velocity_y_curr.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: buf.velocity_y_prev.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: buf.velocity_z_curr.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: buf.velocity_z_prev.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: buf.temperature_curr.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: buf.temperature_prev.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: buf.q_ac.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: buf.config_buffer.as_entire_binding(),
            },
        ],
    })
}
