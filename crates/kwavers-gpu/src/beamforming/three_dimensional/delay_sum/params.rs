//! GPU shader parameters for delay-and-sum and dynamic-focus beamforming.

/// GPU shader parameters for static delay-and-sum (WGSL-compatible layout, 96 bytes).
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

/// GPU shader parameters for dynamic-focus delay-and-sum
/// (`dynamic_focus_3d.wgsl` / `DynamicFocusParams`, 112 bytes).
///
/// Field layout (WGSL-compatible, 16-byte aligned blocks):
/// ```text
/// offset  0: volume_dims   [u32; 3] + _padding1  u32   = 16 B
/// offset 16: voxel_spacing [f32; 3] + _padding2  u32   = 16 B
/// offset 32: num_elements  [u32; 3] + _padding3  u32   = 16 B
/// offset 48: element_spacing[f32;3] + _padding4  u32   = 16 B
/// offset 64: sound_speed, sampling_freq, center_freq, f_number = 16 B
/// offset 80: min_depth, max_depth, num_focus_zones, _padding5  = 16 B
/// offset 96: num_frames, num_samples, enable_variable_aperture, _padding6 = 16 B
/// ```
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct DynamicFocusParams {
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
    pub(super) f_number: f32,
    pub(super) min_depth: f32,
    pub(super) max_depth: f32,
    pub(super) num_focus_zones: u32,
    pub(super) _padding5: u32,
    pub(super) num_frames: u32,
    pub(super) num_samples: u32,
    pub(super) enable_variable_aperture: u32,
    pub(super) _padding6: u32,
}

/// Layout verification — only meaningful in GPU builds where `bytemuck::Pod`
/// requires exact WGSL-compatible alignment.
#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::{DynamicFocusParams, Params};

    #[test]
    fn test_params_layout() {
        assert_eq!(std::mem::size_of::<Params>(), 96);
        assert_eq!(std::mem::align_of::<Params>(), 4);
    }

    #[test]
    fn test_dynamic_focus_params_layout() {
        assert_eq!(std::mem::size_of::<DynamicFocusParams>(), 112);
        assert_eq!(std::mem::align_of::<DynamicFocusParams>(), 4);
    }
}
