//! `encode_velocity_update`: shared FFT(p) cache + per-axis gradient/velocity dispatches.

use super::super::super::{GpuPstdSolver, PstdParams};
use super::StepCtx;

impl GpuPstdSolver {
    /// Encode the velocity-update phase.
    ///
    /// ## Algorithm
    ///
    /// 1. Copy `field_p` → `kspace_re/im` and compute one forward FFT(p).
    /// 2. Save FFT(p) to `absorb_scratch_kre/kim` (reused as a cache register).
    /// 3. For each axis `ax ∈ {0,1,2}`:
    ///    - Restore FFT(p) from cache (skipped for `ax=0` since kspace already holds it).
    ///    - Apply k-space staggered shift in-place on `kspace_re/im`.
    ///    - IFFT → write gradient into `kspace_re`.
    ///    - `vel_update`: accumulate into `ux/uy/uz`.
    ///
    /// Net saving over the naive approach: −2 full 3-D forward FFTs and −8 GPU dispatches
    /// per time step (shared FFT(p) across all three axes).
    pub(super) fn encode_velocity_update(
        &self,
        cpass: &mut wgpu::ComputePass<'_>,
        ctx: &StepCtx,
        bg: &wgpu::BindGroup,
        step: u32,
    ) {
        let ew = ctx.elem_wg;
        self.dispatch(
            cpass,
            &ctx.params(step, 0),
            &self.pipeline_copy_field_to_k,
            bg,
            ew,
            "cp_p",
        );
        self.fft_3d(cpass, bg, step, ctx.n_sensors);
        self.dispatch_absorb(
            cpass,
            &ctx.params(step, 0),
            &self.pipeline_absorb_save_kspace,
            bg,
            ew,
            "save_fftp",
        );

        for ax in 0u32..3u32 {
            if ax > 0 {
                self.dispatch_absorb(
                    cpass,
                    &ctx.params(step, ax),
                    &self.pipeline_absorb_restore_kspace,
                    bg,
                    ew,
                    "rest_fftp",
                );
            }
            self.dispatch(
                cpass,
                &ctx.params(step, ax),
                &self.pipeline_kspace_shift,
                bg,
                ew,
                "kshift_v",
            );
            self.ifft_3d(cpass, bg, step, ctx.n_sensors);
            self.dispatch(
                cpass,
                &ctx.params(step, ax),
                &self.pipeline_vel_update,
                bg,
                ew,
                "vel_upd",
            );
        }
    }
}
