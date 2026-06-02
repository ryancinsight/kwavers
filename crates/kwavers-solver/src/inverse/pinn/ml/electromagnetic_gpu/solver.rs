//! `GPUEMSolver` struct definition.
//!
//! SRP: changes when the solver's field inventory or GPU resource layout changes.

use super::compute::ComputeManager;
use super::config::EMConfig;
use super::fields::EMFieldData;
use std::collections::HashMap;

/// GPU-accelerated electromagnetic solver.
#[derive(Debug)]
pub struct GPUEMSolver {
    pub(super) config: EMConfig,
    pub(super) compute_manager: ComputeManager,
    pub(super) field_data: Option<EMFieldData>,
    pub(super) gpu_buffers: HashMap<String, wgpu::Buffer>,
    pub(super) compute_pipeline: Option<wgpu::ComputePipeline>,
    pub(super) bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub(super) bind_group: Option<wgpu::BindGroup>,
}
