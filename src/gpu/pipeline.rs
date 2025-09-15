//! GPU compute pipeline management

use crate::error::{KwaversError, KwaversResult};
use std::collections::HashMap;

/// Compute pipeline wrapper
#[derive(Debug)]
pub struct ComputePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    workgroup_size: [u32; 3],
}

impl ComputePipeline {
    /// Create a new compute pipeline
    pub fn new(
        device: &wgpu::Device,
        shader_source: &str,
        entry_point: &str,
        workgroup_size: [u32; 3],
    ) -> KwaversResult<Self> {
        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
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

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point,
            compilation_options: Default::default(),
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
            workgroup_size,
        })
    }

    /// Get pipeline reference
    pub fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }

    /// Get bind group layout
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Get workgroup size
    pub fn workgroup_size(&self) -> [u32; 3] {
        self.workgroup_size
    }

    /// Calculate dispatch size for given problem size
    pub fn dispatch_size(&self, problem_size: [u32; 3]) -> [u32; 3] {
        [
            (problem_size[0] + self.workgroup_size[0] - 1) / self.workgroup_size[0],
            (problem_size[1] + self.workgroup_size[1] - 1) / self.workgroup_size[1],
            (problem_size[2] + self.workgroup_size[2] - 1) / self.workgroup_size[2],
        ]
    }
}

/// Pipeline layout manager
#[derive(Debug)]
pub struct PipelineLayout {
    layouts: HashMap<String, wgpu::PipelineLayout>,
}

impl PipelineLayout {
    /// Create a new pipeline layout manager
    pub fn new() -> Self {
        Self {
            layouts: HashMap::new(),
        }
    }

    /// Create and store a pipeline layout
    pub fn create_layout(
        &mut self,
        device: &wgpu::Device,
        name: &str,
        bind_group_layouts: &[&wgpu::BindGroupLayout],
        push_constant_ranges: &[wgpu::PushConstantRange],
    ) -> Result<&wgpu::PipelineLayout, crate::error::KwaversError> {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(name),
            bind_group_layouts,
            push_constant_ranges,
        });

        self.layouts.insert(name.to_string(), layout);
        self.layouts.get(name).ok_or_else(|| {
            crate::error::KwaversError::System(crate::error::SystemError::ResourceExhausted {
                resource: format!("Pipeline layout '{}'", name),
                reason: "Layout not found after creation".to_string(),
            })
        })
    }

    /// Get a pipeline layout by name
    pub fn get(&self, name: &str) -> Option<&wgpu::PipelineLayout> {
        self.layouts.get(name)
    }
}
