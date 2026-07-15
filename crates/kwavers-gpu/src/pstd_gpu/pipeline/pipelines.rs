//! PSTD compute-pipeline provider contracts.

/// Provider contract for PSTD compute-pipeline creation.
pub trait PstdPipelineProvider {
    /// Provider-owned compute-pipeline type.
    type Pipeline;
    /// Provider-owned bind-group-layout type.
    type BindGroupLayout;
    /// Provider-owned pipeline-layout type.
    type PipelineLayout;
    /// Provider-owned shader or kernel module type.
    type ShaderModule;

    /// Create a shader or kernel module from provider-specific source.
    fn shader_module(&self, source: &'static str, label: &'static str) -> Self::ShaderModule;

    /// Create a pipeline layout for PSTD compute kernels.
    fn pipeline_layout(
        &self,
        bind_group_layouts: &[Option<&Self::BindGroupLayout>],
        immediate_data_bytes: usize,
        label: &'static str,
    ) -> Self::PipelineLayout;

    /// Create a compute pipeline for one PSTD kernel entry point.
    fn compute_pipeline(
        &self,
        layout: &Self::PipelineLayout,
        shader: &Self::ShaderModule,
        entry: &'static str,
    ) -> Self::Pipeline;
}

/// WGPU PSTD compute-pipeline provider.
pub struct WgpuPstdPipelineFactory<'a> {
    device: &'a wgpu::Device,
}

impl<'a> WgpuPstdPipelineFactory<'a> {
    /// Create a WGPU PSTD pipeline factory.
    #[must_use]
    pub const fn new(device: &'a wgpu::Device) -> Self {
        Self { device }
    }
}

impl PstdPipelineProvider for WgpuPstdPipelineFactory<'_> {
    type Pipeline = wgpu::ComputePipeline;
    type BindGroupLayout = wgpu::BindGroupLayout;
    type PipelineLayout = wgpu::PipelineLayout;
    type ShaderModule = wgpu::ShaderModule;

    fn shader_module(&self, source: &'static str, label: &'static str) -> Self::ShaderModule {
        self.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            })
    }

    fn pipeline_layout(
        &self,
        bind_group_layouts: &[Option<&Self::BindGroupLayout>],
        immediate_data_bytes: usize,
        label: &'static str,
    ) -> Self::PipelineLayout {
        self.device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts,
                immediate_size: immediate_data_bytes as u32,
            })
    }

    fn compute_pipeline(
        &self,
        layout: &Self::PipelineLayout,
        shader: &Self::ShaderModule,
        entry: &'static str,
    ) -> Self::Pipeline {
        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(layout),
                module: shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
    }
}
