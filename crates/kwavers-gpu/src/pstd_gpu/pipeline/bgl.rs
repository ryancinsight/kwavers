//! PSTD bind-group layout provider contracts.
//!
//! SRP: changes when the bind-group layout changes: binding count, read-only
//! flags, visibility, or group assignment.

/// Provider contract for PSTD bind-group layout creation.
pub trait PstdBindGroupLayoutProvider {
    /// Provider-owned bind-group layout type.
    type BindGroupLayout;

    /// Create group(0): 7 read/write field buffers + 1 read-only source-kappa.
    fn fields_layout(&self) -> Self::BindGroupLayout;

    /// Create group(1): 2 read/write k-space buffers + 6 read-only medium buffers.
    fn kspace_layout(&self) -> Self::BindGroupLayout;

    /// Create group(2): sensor/source data.
    fn sensor_layout(&self) -> Self::BindGroupLayout;

    /// Create group(3): fractional-Laplacian absorption constants and scratch buffers.
    fn absorb_layout(&self) -> Self::BindGroupLayout;
}

/// WGPU PSTD bind-group layout provider.
pub struct WgpuPstdBindGroupLayoutFactory<'a> {
    device: &'a wgpu::Device,
}

impl<'a> WgpuPstdBindGroupLayoutFactory<'a> {
    /// Create a WGPU PSTD bind-group layout factory.
    #[must_use]
    pub const fn new(device: &'a wgpu::Device) -> Self {
        Self { device }
    }

    fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    fn layout(
        &self,
        label: &'static str,
        entries: &[wgpu::BindGroupLayoutEntry],
    ) -> wgpu::BindGroupLayout {
        self.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries,
            })
    }
}

impl PstdBindGroupLayoutProvider for WgpuPstdBindGroupLayoutFactory<'_> {
    type BindGroupLayout = wgpu::BindGroupLayout;

    fn fields_layout(&self) -> Self::BindGroupLayout {
        let rw = |binding| Self::storage_entry(binding, false);
        let ro = |binding| Self::storage_entry(binding, true);
        self.layout(
            "bgl_fields",
            &[rw(0), rw(1), rw(2), rw(3), rw(4), rw(5), rw(6), ro(7)],
        )
    }

    fn kspace_layout(&self) -> Self::BindGroupLayout {
        let rw = |binding| Self::storage_entry(binding, false);
        let ro = |binding| Self::storage_entry(binding, true);
        self.layout(
            "bgl_kspace",
            &[rw(0), rw(1), ro(2), ro(3), ro(4), ro(5), ro(6), ro(7)],
        )
    }

    fn sensor_layout(&self) -> Self::BindGroupLayout {
        let ro = |binding| Self::storage_entry(binding, true);
        let rw = |binding| Self::storage_entry(binding, false);
        self.layout(
            "bgl_sensor",
            &[ro(0), ro(1), ro(2), ro(3), ro(4), ro(5), rw(6), ro(7)],
        )
    }

    fn absorb_layout(&self) -> Self::BindGroupLayout {
        let ro = |binding| Self::storage_entry(binding, true);
        let rw = |binding| Self::storage_entry(binding, false);
        self.layout(
            "bgl_absorb",
            &[ro(0), ro(1), ro(2), ro(3), rw(4), rw(5), rw(6), rw(7)],
        )
    }
}
