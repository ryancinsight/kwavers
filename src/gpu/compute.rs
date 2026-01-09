//! GPU compute operations

#[allow(unused_imports)]
use crate::core::error::KwaversResult;
#[allow(unused_imports)]
use wgpu::util::DeviceExt;

/// GPU compute manager
#[derive(Debug)]
pub struct GpuCompute {
    bind_group_layouts: Vec<wgpu::BindGroupLayout>,
    command_encoder: Option<wgpu::CommandEncoder>,
}

impl GpuCompute {
    /// Create a new compute manager
    pub fn new(device: &wgpu::Device) -> Self {
        // Create common bind group layouts
        let storage_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Storage Layout"),
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
            ],
        });

        Self {
            bind_group_layouts: vec![storage_layout],
            command_encoder: None,
        }
    }

    /// Begin command recording
    pub fn begin_commands(&mut self, device: &wgpu::Device) {
        self.command_encoder = Some(device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Compute Command Encoder"),
            },
        ));
    }

    /// End command recording and return buffer
    pub fn finish_commands(&mut self) -> Option<wgpu::CommandBuffer> {
        self.command_encoder.take().map(|encoder| encoder.finish())
    }

    /// Dispatch compute shader
    pub fn dispatch(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: [u32; 3],
    ) {
        if let Some(ref mut encoder) = self.command_encoder {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
        }
    }

    /// Copy buffer to buffer
    pub fn copy_buffer(&mut self, source: &wgpu::Buffer, destination: &wgpu::Buffer, size: u64) {
        if let Some(ref mut encoder) = self.command_encoder {
            encoder.copy_buffer_to_buffer(source, 0, destination, 0, size);
        }
    }

    /// Get bind group layout
    pub fn get_bind_group_layout(&self, index: usize) -> Option<&wgpu::BindGroupLayout> {
        self.bind_group_layouts.get(index)
    }
}
