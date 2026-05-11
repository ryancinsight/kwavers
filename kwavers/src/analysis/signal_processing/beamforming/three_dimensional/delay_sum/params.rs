//! GPU shader parameters for delay-and-sum beamforming.

/// GPU shader parameters (WGSL-compatible layout)
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct Params {
    pub(super) volume_dims: [u32; 3],
    pub(super) _padding1: u32,
    pub(super) voxel_spacing: [f32; 3],
    pub(super) _padding2: u32,
    pub(super) num_elements: [u32; 3],
    pub(super) _padding3: u32,
    pub(super) element_spacing: [f32; 3],
    pub(super) _padding4: u32,
    pub(super) sound_speed: f32,
    pub(super) sampling_freq: f32,
    pub(super) center_freq: f32,
    pub(super) _padding5: f32,
    pub(super) num_frames: u32,
    pub(super) num_samples: u32,
    pub(super) dynamic_focusing: u32,
    pub(super) apodization_window: u32,
}

/// Layout verification — only meaningful in GPU builds where `bytemuck::Pod`
/// requires exact WGSL-compatible alignment.
#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::Params;

    #[test]
    fn test_params_layout() {
        assert_eq!(std::mem::size_of::<Params>(), 96);
        assert_eq!(std::mem::align_of::<Params>(), 4);
    }
}
