//! GPU uniform buffer layout for the MVDR 3D compute shader.

/// Uniform parameters for `mvdr_3d.wgsl`.
///
/// ## Layout guarantee
/// 24 scalar (u32/f32) fields at 4-byte natural alignment → 96 bytes total
/// (6 × 16-byte uniform rows).  The layout must match `struct MvdrParams` in
/// `mvdr_3d.wgsl` field-for-field.  `#[repr(C)]` and `bytemuck::Pod` enforce
/// ABI compatibility; any field reordering is a correctness defect.
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct MvdrGpuParams {
    // Row 0 (bytes 0–15)
    pub vol_x: u32,
    pub vol_y: u32,
    pub vol_z: u32,
    pub nel_x: u32,
    // Row 1 (bytes 16–31)
    pub nel_y: u32,
    pub nel_z: u32,
    pub sub_x: u32,
    pub sub_y: u32,
    // Row 2 (bytes 32–47)
    pub sub_z: u32,
    pub num_frames: u32,
    pub num_samples: u32,
    pub pad0: u32,
    // Row 3 (bytes 48–63)
    pub vox_dx: f32,
    pub vox_dy: f32,
    pub vox_dz: f32,
    pub elem_sx: f32,
    // Row 4 (bytes 64–79)
    pub elem_sy: f32,
    pub elem_sz: f32,
    pub sound_speed: f32,
    pub sampling_freq: f32,
    // Row 5 (bytes 80–95)
    pub diagonal_loading: f32,
    pub pad1: f32,
    pub pad2: f32,
    pub pad3: f32,
}
