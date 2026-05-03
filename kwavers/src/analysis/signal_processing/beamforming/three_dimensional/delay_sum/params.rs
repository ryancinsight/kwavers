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

/// CPU fallback params (when GPU feature is disabled)
#[cfg(not(feature = "gpu"))]
#[repr(C)]
#[derive(Copy, Clone)]
#[allow(dead_code)]
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

#[cfg(test)]
mod tests {
    use super::Params;

    #[test]
    fn test_params_layout() {
        // Verify that Params struct has correct alignment for GPU
        #[cfg(feature = "gpu")]
        assert_eq!(std::mem::size_of::<Params>(), 96);
        #[cfg(not(feature = "gpu"))]
        assert_eq!(std::mem::size_of::<Params>(), 96);
        assert_eq!(std::mem::align_of::<Params>(), 4);
    }
}
