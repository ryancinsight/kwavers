//! Bind group layout builders for `GpuPstdSolver`.
//!
//! SRP: changes when the bind group layout changes (number of bindings,
//! read_only flags, or group assignment).

/// group(0): 7 read_write field buffers + 1 read-only source-kappa.
pub(super) fn build_bgl_fields(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let rw = |b: u32| wgpu::BindGroupLayoutEntry {
        binding: b,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let ro = |b: u32| wgpu::BindGroupLayoutEntry {
        binding: b,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl_fields"),
        entries: &[rw(0), rw(1), rw(2), rw(3), rw(4), rw(5), rw(6), ro(7)],
    })
}

/// group(1): 2 read_write k-space buffers + 6 read-only medium buffers.
///
/// Bindings: kspace_re (0), kspace_im (1), kappa (2), rho0_inv (3),
/// c0_sq (4), rho0 (5), bon_a (6), alpha_decay/twiddle (7).
pub(super) fn build_bgl_kspace(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let rw = |b: u32| wgpu::BindGroupLayoutEntry {
        binding: b,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let ro = |b: u32| wgpu::BindGroupLayoutEntry {
        binding: b,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl_kspace"),
        entries: &[rw(0), rw(1), ro(2), ro(3), ro(4), ro(5), ro(6), ro(7)],
    })
}

/// group(2): sensor/source data — 5 read-only + 1 read_write sensor_data
/// + 1 read-only sensor_indices + 1 read-only source_data.
///
/// Bindings: pml_sgx (0), pml_sgy (1), pml_sgz (2), pml_xyz (3),
/// shifts_all (4), sensor_flat_indices (5), sensor_data rw (6), source_data (7).
pub(super) fn build_bgl_sensor(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let ro = |b: u32| wgpu::BindGroupLayoutEntry {
        binding: b,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl_sensor"),
        entries: &[
            ro(0),
            ro(1),
            ro(2),
            ro(3),
            ro(4),
            ro(5),
            // binding 6: sensor_data (read_write)
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            ro(7),
        ],
    })
}

/// group(3): fractional-Laplacian absorption — 4 read-only constants +
/// 4 read_write scratch buffers.
///
/// Bindings: nabla1 (0), nabla2 (1), tau (2), eta (3),
/// scratch_kre (4), scratch_kim (5), l1 (6), l2 (7).
pub(super) fn build_bgl_absorb(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let ro = |b: u32| wgpu::BindGroupLayoutEntry {
        binding: b,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let rw = |b: u32| wgpu::BindGroupLayoutEntry {
        binding: b,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl_absorb"),
        entries: &[ro(0), ro(1), ro(2), ro(3), rw(4), rw(5), rw(6), rw(7)],
    })
}
