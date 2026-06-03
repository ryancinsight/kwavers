//! Bind group assembly for `GpuPstdSolver`.
//!
//! SRP: changes when the binding slot assignment within a group changes.
//!
//! Each builder takes a small `*Buffers` struct (rather than a long positional
//! buffer list) so call sites name every binding explicitly.

/// Group(0) field buffers, in binding-slot order.
pub(super) struct FieldBuffers<'a> {
    pub p: &'a wgpu::Buffer,
    pub ux: &'a wgpu::Buffer,
    pub uy: &'a wgpu::Buffer,
    pub uz: &'a wgpu::Buffer,
    pub rhox: &'a wgpu::Buffer,
    pub rhoy: &'a wgpu::Buffer,
    pub rhoz: &'a wgpu::Buffer,
    pub source_kappa: &'a wgpu::Buffer,
}

/// Group(1) k-space + medium buffers, in binding-slot order.
pub(super) struct KspaceBuffers<'a> {
    pub kspace_re: &'a wgpu::Buffer,
    pub kspace_im: &'a wgpu::Buffer,
    pub kappa: &'a wgpu::Buffer,
    pub rho0_inv: &'a wgpu::Buffer,
    pub c0_sq: &'a wgpu::Buffer,
    pub rho0: &'a wgpu::Buffer,
    pub bon_a: &'a wgpu::Buffer,
    pub alpha_decay: &'a wgpu::Buffer,
}

/// Group(2) absorption operator + scratch buffers, in binding-slot order.
pub(super) struct AbsorbBuffers<'a> {
    pub nabla1: &'a wgpu::Buffer,
    pub nabla2: &'a wgpu::Buffer,
    pub tau: &'a wgpu::Buffer,
    pub eta: &'a wgpu::Buffer,
    pub scratch_kre: &'a wgpu::Buffer,
    pub scratch_kim: &'a wgpu::Buffer,
    pub scratch_l1: &'a wgpu::Buffer,
    pub scratch_l2: &'a wgpu::Buffer,
}

/// Assemble an 8-binding bind group whose slots are filled in array order.
fn bind_group(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    buffers: [&wgpu::Buffer; 8],
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &entries,
    })
}

/// Build group(0): field buffers (p, ux, uy, uz, rhox, rhoy, rhoz, source_kappa).
pub(super) fn build_bg_fields(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    f: &FieldBuffers,
) -> wgpu::BindGroup {
    bind_group(
        device,
        "bg_fields",
        layout,
        [
            f.p,
            f.ux,
            f.uy,
            f.uz,
            f.rhox,
            f.rhoy,
            f.rhoz,
            f.source_kappa,
        ],
    )
}

/// Build group(1): k-space + medium (kspace_re, kspace_im, kappa, rho0_inv,
/// c0_sq, rho0, bon_a, alpha_decay).
pub(super) fn build_bg_kspace(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    k: &KspaceBuffers,
) -> wgpu::BindGroup {
    bind_group(
        device,
        "bg_kspace",
        layout,
        [
            k.kspace_re,
            k.kspace_im,
            k.kappa,
            k.rho0_inv,
            k.c0_sq,
            k.rho0,
            k.bon_a,
            k.alpha_decay,
        ],
    )
}

/// Build group(2): absorption operators + scratch (nabla1, nabla2, tau, eta,
/// scratch_kre, scratch_kim, scratch_l1, scratch_l2).
pub(super) fn build_bg_absorb(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    a: &AbsorbBuffers,
) -> wgpu::BindGroup {
    bind_group(
        device,
        "bg_absorb",
        layout,
        [
            a.nabla1,
            a.nabla2,
            a.tau,
            a.eta,
            a.scratch_kre,
            a.scratch_kim,
            a.scratch_l1,
            a.scratch_l2,
        ],
    )
}
