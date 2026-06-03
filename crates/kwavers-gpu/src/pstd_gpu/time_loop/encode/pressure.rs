//! `encode_pressure_record`: pressure-from-density, absorption correction, sensor recording.

use super::super::super::{GpuPstdSolver, PstdParams};
use super::StepCtx;

impl GpuPstdSolver {
    /// Encode pressure-from-density, absorption correction, and sensor recording.
    ///
    /// `pres_density` accumulates `rhox + rhoy + rhoz` into `field_p`.  When
    /// absorbing, `absorb_pressure_correction` adds `c₀² · (τ·L1 − η·L2)`.
    /// `record` scatters `field_p[sensor_indices]` into the sensor_data buffer.
    pub(in crate::pstd_gpu) fn encode_pressure_record(
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
            &self.pipeline_pres_density,
            bg,
            ew,
            "pres",
        );
        if ctx.absorbing != 0 {
            self.dispatch_absorb(
                cpass,
                &ctx.params(step, 0),
                &self.pipeline_absorb_pressure_correction,
                bg,
                ew,
                "abs_pres_corr",
            );
        }
        if ctx.n_sensors > 0 {
            self.dispatch(
                cpass,
                &ctx.params(step, 0),
                &self.pipeline_record,
                bg,
                StepCtx::ceil_div(ctx.n_sensors as usize, 256),
                "rec",
            );
        }
    }
}
