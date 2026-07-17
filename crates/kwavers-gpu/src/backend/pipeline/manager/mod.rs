//! WGPU pipeline manager for compute shader compilation and execution.

use super::types::PipelineType;
use kwavers_core::error::KwaversResult;
use std::collections::HashMap;
use wgpu;

mod compile;
mod execute;

/// WGPU pipeline manager for WGSL compute shader execution.
#[derive(Debug)]
pub struct WgpuPipelineManager {
    pub(super) pipelines: HashMap<PipelineType, wgpu::ComputePipeline>,
    pub(super) layouts: HashMap<PipelineType, wgpu::BindGroupLayout>,
}

impl WgpuPipelineManager {
    /// Create a new WGPU pipeline manager and compile all shaders.
    /// # Errors
    /// - Propagates errors returned by called functions.
    ///
    pub fn new(device: &wgpu::Device) -> KwaversResult<Self> {
        let mut pipelines = HashMap::new();
        let mut layouts = HashMap::new();

        Self::compile_elementwise_pipeline(device, &mut pipelines, &mut layouts)?;
        Self::compile_derivative_pipeline(device, &mut pipelines, &mut layouts)?;

        Ok(Self { pipelines, layouts })
    }
}
