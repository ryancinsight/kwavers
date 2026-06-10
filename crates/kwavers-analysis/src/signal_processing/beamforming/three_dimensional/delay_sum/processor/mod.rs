//! GPU delay-and-sum beamforming processor.

mod dispatch;
pub(crate) mod dynamic_focus_dispatch;

#[cfg(feature = "gpu")]
use crate::signal_processing::beamforming::three_dimensional::config::BeamformingConfig3D;

/// GPU delay-and-sum beamforming implementation.
///
/// Only available when the `gpu` feature is enabled. The `BeamformingProcessor3D::new()`
/// constructor returns `Err` in non-GPU builds, so this type is never instantiated there.
#[cfg(feature = "gpu")]
pub struct DelaySumGPU<'a> {
    pub(super) config: &'a BeamformingConfig3D,
    pub(super) device: &'a wgpu::Device,
    pub(super) queue: &'a wgpu::Queue,
    pub(super) pipeline: &'a wgpu::ComputePipeline,
    pub(super) bind_group_layout: &'a wgpu::BindGroupLayout,
}

#[cfg(feature = "gpu")]
impl<'a> DelaySumGPU<'a> {
    /// Create new GPU delay-and-sum processor.
    pub fn new(
        config: &'a BeamformingConfig3D,
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        pipeline: &'a wgpu::ComputePipeline,
        bind_group_layout: &'a wgpu::BindGroupLayout,
    ) -> Self {
        Self {
            config,
            device,
            queue,
            pipeline,
            bind_group_layout,
        }
    }

    /// Create element position array for the 3D transducer aperture.
    ///
    /// Returns flat `[x₀, y₀, z₀, x₁, y₁, z₁, …]` for the GPU buffer.
    pub(super) fn create_element_positions(&self) -> Vec<f32> {
        let mut element_positions = Vec::new();
        for ex in 0..self.config.num_elements_3d.0 {
            for ey in 0..self.config.num_elements_3d.1 {
                for ez in 0..self.config.num_elements_3d.2 {
                    let x = ((self.config.num_elements_3d.0 - 1) as f32).mul_add(-0.5, ex as f32)
                        * self.config.element_spacing_3d.0 as f32;
                    let y = ((self.config.num_elements_3d.1 - 1) as f32).mul_add(-0.5, ey as f32)
                        * self.config.element_spacing_3d.1 as f32;
                    let z = ((self.config.num_elements_3d.2 - 1) as f32).mul_add(-0.5, ez as f32)
                        * self.config.element_spacing_3d.2 as f32;
                    element_positions.extend_from_slice(&[x, y, z]);
                }
            }
        }
        element_positions
    }
}
