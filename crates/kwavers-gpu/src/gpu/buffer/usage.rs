/// Buffer usage flags for GPU buffers
///
/// These constants define how a buffer will be used, which affects memory allocation
/// and allowed operations. Multiple flags can be combined using the bitwise OR operator.
///
/// # Examples
///
/// ```
/// # use kwavers::gpu::buffer::BufferUsage;
/// // Storage buffer that can be copied from
/// let usage = BufferUsage::STORAGE | BufferUsage::COPY_SRC;
/// ```
#[derive(Debug)]
pub struct BufferUsage;

impl BufferUsage {
    /// Storage buffer usage - can be bound as storage buffer in shaders
    pub const STORAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE;

    /// Uniform buffer usage - can be bound as uniform buffer in shaders
    pub const UNIFORM: wgpu::BufferUsages = wgpu::BufferUsages::UNIFORM;

    /// Copy source - can be used as source in copy operations
    pub const COPY_SRC: wgpu::BufferUsages = wgpu::BufferUsages::COPY_SRC;

    /// Copy destination - can be used as destination in copy operations
    pub const COPY_DST: wgpu::BufferUsages = wgpu::BufferUsages::COPY_DST;
}
