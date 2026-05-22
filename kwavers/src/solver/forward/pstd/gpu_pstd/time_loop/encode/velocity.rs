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
    ///    - **ax=0**: kspace already holds FFT(p); apply `kspace_shift_apply` in-place.
    ///    - **ax=1,2**: call `restore_and_shift_apply` — reads FFT(p) from
    ///      `absorb_scratch_kre/kim`, applies kappa × shift, writes directly to
    ///      `kspace_re/im`. This fuses the old `absorb_restore_kspace` + `kspace_shift`
    ///      two-dispatch sequence into one, saving 4N f32 of global memory traffic and
    ///      one GPU dispatch per axis (−2 dispatches/step total).
    ///    - IFFT → gradient in `kspace_re`.
    ///    - `velocity_update`: accumulate PML-damped gradient into `ux/uy/uz`.
    ///
    /// ## Traffic accounting (per step, element count = N = nx×ny×nz)
    ///
    /// | Operation                        | Dispatches | Memory traffic |
    /// |----------------------------------|-----------|---------------|
    /// | copy_field_to_kspace             | 1         | 3N f32        |
    /// | fft_3d                           | 3         | in SMEM       |
    /// | absorb_save_kspace               | 1         | 4N f32        |
    /// | ax=0: kspace_shift               | 1         | 4N f32        |
    /// | ax=1,2: restore_and_shift (each) | 1         | 5N f32 (was 8N) |
    /// | ifft_3d × 3 axes                 | 9         | in SMEM       |
    /// | velocity_update × 3 axes         | 3         | 4N f32 each   |
    /// | **Total per step**               | **19**    | **vs 21 before** |
    ///
    /// Net over naive (no shared FFT(p)): −2 full 3-D forward FFTs per step.
    pub(in crate::solver::forward::pstd::gpu_pstd) fn encode_velocity_update(
        &self,
        cpass: &mut wgpu::ComputePass<'_>,
        ctx: &StepCtx,
        bg: &wgpu::BindGroup,
        step: u32,
    ) {
        let ew = ctx.elem_wg;

        // Step 1: copy field_p → kspace, compute one shared forward FFT(p).
        self.dispatch(
            cpass,
            &ctx.params(step, 0),
            &self.pipeline_copy_field_to_k,
            bg,
            ew,
            "cp_p",
        );
        self.fft_3d(cpass, bg, step, ctx.n_sensors);

        // Step 2: cache FFT(p) into absorb_scratch so axes 1 and 2 can read it.
        self.dispatch_absorb(
            cpass,
            &ctx.params(step, 0),
            &self.pipeline_absorb_save_kspace,
            bg,
            ew,
            "save_fftp",
        );

        // Step 3: per-axis gradient → velocity update.
        for ax in 0u32..3u32 {
            if ax == 0 {
                // kspace already holds FFT(p) from the shared FFT above;
                // apply shift in-place (standard 3-group pipeline, no absorb group).
                self.dispatch(
                    cpass,
                    &ctx.params(step, ax),
                    &self.pipeline_kspace_shift,
                    bg,
                    ew,
                    "kshift_v0",
                );
            } else {
                // Fused restore + shift: reads FFT(p) from absorb_scratch,
                // applies kappa × shift operator, writes to kspace_re/im.
                // Saves 1 dispatch + 4N f32 vs. separate restore + shift.
                self.dispatch_absorb(
                    cpass,
                    &ctx.params(step, ax),
                    &self.pipeline_restore_and_shift,
                    bg,
                    ew,
                    "restore_shift",
                );
            }
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
