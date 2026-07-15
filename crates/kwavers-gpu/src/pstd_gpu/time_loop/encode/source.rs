//! `encode_source_injection`: pressure-source and velocity-source injection.

use super::super::super::{state::WgpuPstdState, PstdParams};
use super::StepCtx;

impl WgpuPstdState {
    /// Encode pressure-source and velocity-source injection.
    ///
    /// Pressure injection: scatter `source_signals[step]` to `field_p[source_indices]`.
    ///
    /// Velocity-source injection (when `ctx.n_vel_x > 0`):
    /// zeros `kspace_re/im`, scatters the velocity signal to kspace, applies
    /// `source_kappa` k-correction via FFT/IFFT, then adds the corrected field
    /// back into `ux`.  This avoids the PCIe-side `clear_buffer` which would end
    /// the compute pass; the GPU `pipeline_zero_kspace` shader clears in-pass.
    pub(in crate::pstd_gpu) fn encode_source_injection(
        &self,
        cpass: &mut wgpu::ComputePass<'_>,
        ctx: &StepCtx,
        bg: &wgpu::BindGroup,
        bg_vel: &wgpu::BindGroup,
        step: u32,
    ) {
        if ctx.n_src > 0 {
            let src_params = PstdParams {
                axis: ctx.n_src as u32,
                ..ctx.params(step, 0)
            };
            self.dispatch(
                cpass,
                &src_params,
                &self.pipelines.inject_src,
                bg,
                StepCtx::ceil_div(ctx.n_src, 256),
                "inject",
            );
        }

        if ctx.n_vel_x > 0 {
            let ew = ctx.elem_wg;
            self.dispatch(
                cpass,
                &ctx.params(step, 0),
                &self.pipelines.zero_kspace,
                bg,
                ew,
                "zero_ksp",
            );
            let vel_params = PstdParams {
                axis: ctx.n_vel_x as u32,
                ..ctx.params(step, 0)
            };
            self.dispatch(
                cpass,
                &vel_params,
                &self.pipelines.inject_vel_x,
                bg_vel,
                StepCtx::ceil_div(ctx.n_vel_x, 256),
                "inject_vx",
            );
            self.fft_3d(cpass, bg, ctx, step);
            self.dispatch(
                cpass,
                &ctx.params(step, 0),
                &self.pipelines.apply_source_kappa,
                bg,
                ew,
                "src_kappa",
            );
            self.ifft_3d(cpass, bg, ctx, step);
            self.dispatch(
                cpass,
                &ctx.params(step, 0),
                &self.pipelines.add_kspace_to_field_ux,
                bg,
                ew,
                "add_ux",
            );
        }
    }
}
