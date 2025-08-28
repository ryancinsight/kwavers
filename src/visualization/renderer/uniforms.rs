//! Uniform data structures for GPU rendering

/// Volume rendering uniforms
#[repr(C)]
#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "gpu-visualization", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct VolumeUniforms {
    /// View matrix
    pub view_matrix: [[f32; 4]; 4],
    /// Projection matrix
    pub projection_matrix: [[f32; 4]; 4],
    /// Volume size
    pub volume_size: [f32; 3],
    /// Padding for alignment
    pub _padding: f32,
    /// Ray step size
    pub ray_step: f32,
    /// Density scale
    pub density_scale: f32,
    /// Brightness
    pub brightness: f32,
    /// Contrast
    pub contrast: f32,
}

impl Default for VolumeUniforms {
    fn default() -> Self {
        Self {
            view_matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            projection_matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            volume_size: [1.0, 1.0, 1.0],
            _padding: 0.0,
            ray_step: 0.01,
            density_scale: 1.0,
            brightness: 1.0,
            contrast: 1.0,
        }
    }
}