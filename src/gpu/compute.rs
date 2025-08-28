//! Compute pipeline management

use super::GpuBuffer;
use crate::KwaversResult;

/// Compute pipeline for GPU kernels
pub struct ComputePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    name: String,
}

impl ComputePipeline {
    /// Create compute pipeline from shader
    pub fn create(
        device: &wgpu::Device,
        name: &str,
        source: &str,
        entry_point: &str,
    ) -> KwaversResult<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{}_bind_group", name)),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{}_layout", name)),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
            name: name.to_string(),
        })
    }

    /// Dispatch compute work
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        workgroups: (u32, u32, u32),
        buffers: &[&GpuBuffer],
    ) -> KwaversResult<()> {
        if buffers.is_empty() {
            return Err(crate::KwaversError::Config(
                crate::ConfigError::InvalidValue {
                    field: "buffers".to_string(),
                    value: "empty".to_string(),
                    expected: "At least one buffer required".to_string(),
                },
            ));
        }

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{}_bind_group", self.name)),
            layout: &self.bind_group_layout,
            entries: &buffers
                .iter()
                .enumerate()
                .map(|(i, buffer)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: buffer.buffer().as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("{}_encoder", self.name)),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{}_pass", self.name)),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }

        queue.submit(Some(encoder.finish()));

        Ok(())
    }
}
