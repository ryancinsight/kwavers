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
    pub twiddle_fft: &'a wgpu::Buffer,
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

/// Provider contract for PSTD bind-group assembly.
pub trait PstdBindGroupProvider {
    /// Provider-owned buffer type.
    type Buffer;
    /// Provider-owned bind-group-layout type.
    type BindGroupLayout;
    /// Provider-owned bind-group type.
    type BindGroup;

    /// Assemble an 8-binding bind group whose slots are filled in array order.
    fn bind_group(
        &self,
        label: &'static str,
        layout: &Self::BindGroupLayout,
        buffers: [&Self::Buffer; 8],
    ) -> Self::BindGroup;
}

/// WGPU PSTD bind-group provider.
pub struct WgpuPstdBindGroupFactory<'a> {
    device: &'a wgpu::Device,
}

impl<'a> WgpuPstdBindGroupFactory<'a> {
    /// Create a WGPU PSTD bind-group factory.
    #[must_use]
    pub const fn new(device: &'a wgpu::Device) -> Self {
        Self { device }
    }
}

impl PstdBindGroupProvider for WgpuPstdBindGroupFactory<'_> {
    type Buffer = wgpu::Buffer;
    type BindGroupLayout = wgpu::BindGroupLayout;
    type BindGroup = wgpu::BindGroup;

    fn bind_group(
        &self,
        label: &'static str,
        layout: &Self::BindGroupLayout,
        buffers: [&Self::Buffer; 8],
    ) -> Self::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &entries,
        })
    }
}

/// Build group(0): field buffers (p, ux, uy, uz, rhox, rhoy, rhoz, source_kappa).
pub(super) fn build_bg_fields(
    provider: &impl PstdBindGroupProvider<
        Buffer = wgpu::Buffer,
        BindGroup = wgpu::BindGroup,
        BindGroupLayout = wgpu::BindGroupLayout,
    >,
    layout: &wgpu::BindGroupLayout,
    f: &FieldBuffers,
) -> wgpu::BindGroup {
    provider.bind_group(
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
/// c0_sq, rho0, bon_a, twiddle_fft).
pub(super) fn build_bg_kspace(
    provider: &impl PstdBindGroupProvider<
        Buffer = wgpu::Buffer,
        BindGroup = wgpu::BindGroup,
        BindGroupLayout = wgpu::BindGroupLayout,
    >,
    layout: &wgpu::BindGroupLayout,
    k: &KspaceBuffers,
) -> wgpu::BindGroup {
    provider.bind_group(
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
            k.twiddle_fft,
        ],
    )
}

/// Build group(2): absorption operators + scratch (nabla1, nabla2, tau, eta,
/// scratch_kre, scratch_kim, scratch_l1, scratch_l2).
pub(super) fn build_bg_absorb(
    provider: &impl PstdBindGroupProvider<
        Buffer = wgpu::Buffer,
        BindGroup = wgpu::BindGroup,
        BindGroupLayout = wgpu::BindGroupLayout,
    >,
    layout: &wgpu::BindGroupLayout,
    a: &AbsorbBuffers,
) -> wgpu::BindGroup {
    provider.bind_group(
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
