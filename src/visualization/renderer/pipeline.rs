//! Rendering and compute pipelines

use crate::error::{KwaversError, KwaversResult};

/// Render pipeline for visualization
#[derive(Debug)]
pub struct RenderPipeline {
    #[cfg(feature = "gpu-visualization")]
    pipeline: Option<wgpu::RenderPipeline>,
    #[cfg(feature = "gpu-visualization")]
    layout: Option<wgpu::PipelineLayout>,
}

impl RenderPipeline {
    /// Create a new render pipeline
    pub fn new() -> KwaversResult<Self> {
        Ok(Self {
            #[cfg(feature = "gpu-visualization")]
            pipeline: None,
            #[cfg(feature = "gpu-visualization")]
            layout: None,
        })
    }

    /// Initialize the pipeline with a device
    #[cfg(feature = "gpu-visualization")]
    pub fn initialize(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
    ) -> KwaversResult<()> {
        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Render Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/render.wgsl").into()),
        });

        // Create pipeline layout
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        // Create render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        self.pipeline = Some(pipeline);
        self.layout = Some(layout);

        Ok(())
    }

    /// Get the pipeline
    #[cfg(feature = "gpu-visualization")]
    pub fn pipeline(&self) -> Option<&wgpu::RenderPipeline> {
        self.pipeline.as_ref()
    }
}

/// Compute pipeline for GPU acceleration
#[derive(Debug)]
pub struct ComputePipeline {
    #[cfg(feature = "gpu-visualization")]
    pipeline: Option<wgpu::ComputePipeline>,
    #[cfg(feature = "gpu-visualization")]
    layout: Option<wgpu::PipelineLayout>,
}

impl ComputePipeline {
    /// Create a new compute pipeline
    pub fn new() -> KwaversResult<Self> {
        Ok(Self {
            #[cfg(feature = "gpu-visualization")]
            pipeline: None,
            #[cfg(feature = "gpu-visualization")]
            layout: None,
        })
    }

    /// Initialize the pipeline with a device
    #[cfg(feature = "gpu-visualization")]
    pub fn initialize(
        &mut self,
        device: &wgpu::Device,
        shader_source: &str,
        entry_point: &str,
    ) -> KwaversResult<()> {
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
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point,
            compilation_options: Default::default(),
        });

        self.pipeline = Some(pipeline);
        self.layout = Some(layout);

        Ok(())
    }

    /// Get the pipeline
    #[cfg(feature = "gpu-visualization")]
    pub fn pipeline(&self) -> Option<&wgpu::ComputePipeline> {
        self.pipeline.as_ref()
    }
}
