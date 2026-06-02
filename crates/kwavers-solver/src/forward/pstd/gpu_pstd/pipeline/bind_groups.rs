//! Bind group assembly for `GpuPstdSolver`.
//!
//! SRP: changes when the binding slot assignment within a group changes.

/// Build group(0): field buffers (p, ux, uy, uz, rhox, rhoy, rhoz, source_kappa).
#[allow(clippy::too_many_arguments)]
pub(super) fn build_bg_fields(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buf_p: &wgpu::Buffer,
    buf_ux: &wgpu::Buffer,
    buf_uy: &wgpu::Buffer,
    buf_uz: &wgpu::Buffer,
    buf_rhox: &wgpu::Buffer,
    buf_rhoy: &wgpu::Buffer,
    buf_rhoz: &wgpu::Buffer,
    buf_source_kappa: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg_fields"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_p.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_ux.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_uy.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buf_uz.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: buf_rhox.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: buf_rhoy.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: buf_rhoz.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: buf_source_kappa.as_entire_binding(),
            },
        ],
    })
}

/// Build group(1): k-space + medium (kspace_re, kspace_im, kappa, rho0_inv,
/// c0_sq, rho0, bon_a, alpha_decay/twiddle).
#[allow(clippy::too_many_arguments)]
pub(super) fn build_bg_kspace(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buf_kspace_re: &wgpu::Buffer,
    buf_kspace_im: &wgpu::Buffer,
    buf_kappa: &wgpu::Buffer,
    buf_rho0_inv: &wgpu::Buffer,
    buf_c0_sq: &wgpu::Buffer,
    buf_rho0: &wgpu::Buffer,
    buf_bon_a: &wgpu::Buffer,
    buf_alpha_decay: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg_kspace"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_kspace_re.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_kspace_im.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_kappa.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buf_rho0_inv.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: buf_c0_sq.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: buf_rho0.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: buf_bon_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: buf_alpha_decay.as_entire_binding(),
            },
        ],
    })
}

/// Build group(3): fractional-Laplacian absorption (nabla1, nabla2, tau, eta,
/// scratch_kre, scratch_kim, l1, l2).
#[allow(clippy::too_many_arguments)]
pub(super) fn build_bg_absorb(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buf_absorb_nabla1: &wgpu::Buffer,
    buf_absorb_nabla2: &wgpu::Buffer,
    buf_absorb_tau: &wgpu::Buffer,
    buf_absorb_eta: &wgpu::Buffer,
    buf_absorb_scratch_kre: &wgpu::Buffer,
    buf_absorb_scratch_kim: &wgpu::Buffer,
    buf_absorb_scratch_l1: &wgpu::Buffer,
    buf_absorb_scratch_l2: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg_absorb"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_absorb_nabla1.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_absorb_nabla2.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_absorb_tau.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buf_absorb_eta.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: buf_absorb_scratch_kre.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: buf_absorb_scratch_kim.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: buf_absorb_scratch_l1.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: buf_absorb_scratch_l2.as_entire_binding(),
            },
        ],
    })
}
