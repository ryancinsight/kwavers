//! Low-level physics dispatch helpers and prepared FFT composition.
//!
//! All methods operate on one Hephaestus grouped sequence so adjacent Kwavers
//! physics kernels and prepared FFT stages share one WGPU compute pass.

use super::super::{state::WgpuPstdState, PstdParams};
use hephaestus_core::Result;
use hephaestus_wgpu::WgpuGroupedSequence;

impl WgpuPstdState {
    /// Encode one dispatch into `cpass` (3-group pipeline layout).
    ///
    /// Immediate data carries `params` inline — no `write_buffer()` per dispatch.
    /// Bind groups: fields(0), kspace+medium(1), sensor(2).
    #[inline]
    pub(super) fn dispatch(
        &self,
        sequence: &mut WgpuGroupedSequence<'_>,
        params: &PstdParams,
        pipeline: &wgpu::ComputePipeline,
        bg_sensor: &wgpu::BindGroup,
        workgroups: u32,
        _label: &str,
    ) {
        let cpass = sequence.raw_pass_mut();
        cpass.set_pipeline(pipeline);
        cpass.set_immediates(0, bytemuck::bytes_of(params));
        cpass.set_bind_group(0, &self.permanent_bind_groups.fields, &[]);
        cpass.set_bind_group(1, &self.permanent_bind_groups.kspace, &[]);
        cpass.set_bind_group(2, bg_sensor, &[]);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    /// Encode a dispatch that also binds the absorption group(3) (4-group layout).
    ///
    /// Used by fractional-Laplacian absorption shaders. The shared pipeline
    /// layout still requires group(2); the absorption kernels do not read it.
    /// Construction creates group(3) only for an absorption-enabled solver.
    #[inline]
    pub(super) fn dispatch_absorb(
        &self,
        sequence: &mut WgpuGroupedSequence<'_>,
        params: &PstdParams,
        pipeline: &wgpu::ComputePipeline,
        bg_sensor: &wgpu::BindGroup,
        workgroups: u32,
        _label: &str,
    ) {
        let cpass = sequence.raw_pass_mut();
        cpass.set_pipeline(pipeline);
        cpass.set_immediates(0, bytemuck::bytes_of(params));
        cpass.set_bind_group(0, &self.permanent_bind_groups.fields, &[]);
        cpass.set_bind_group(1, &self.permanent_bind_groups.kspace, &[]);
        cpass.set_bind_group(2, bg_sensor, &[]);
        let absorption = self.permanent_bind_groups.absorb.as_ref().expect(
            "invariant: absorption dispatch requires an absorption-enabled PSTD bind group",
        );
        cpass.set_bind_group(3, absorption, &[]);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    /// Encode the prepared forward three-dimensional transform.
    pub(super) fn encode_forward_fft(&self, sequence: &mut WgpuGroupedSequence<'_>) -> Result<()> {
        self.fft_plans.forward.encode_in_sequence(sequence)
    }

    /// Encode the prepared, full-volume-normalized inverse transform.
    pub(super) fn encode_inverse_fft(&self, sequence: &mut WgpuGroupedSequence<'_>) -> Result<()> {
        self.fft_plans.inverse.encode_in_sequence(sequence)
    }
}

#[cfg(test)]
#[path = "dispatch_tests.rs"]
mod tests;
