//! `encode_source_injection`: pressure-source and velocity-source injection.

use super::super::super::{state::WgpuPstdState, PstdParams};
use super::StepCtx;

struct SourceInjection<'a> {
    bind_group: &'a wgpu::BindGroup,
    count: usize,
    apply_correction: bool,
    inject_pipeline: &'a wgpu::ComputePipeline,
    add_pipeline: &'a wgpu::ComputePipeline,
}

impl WgpuPstdState {
    /// Encode a k-space-filtered additive source into its destination field.
    fn encode_source_injection(
        &self,
        cpass: &mut wgpu::ComputePass<'_>,
        ctx: &StepCtx,
        bg: &wgpu::BindGroup,
        step: u32,
        injection: SourceInjection<'_>,
    ) {
        let ew = ctx.elem_wg;
        self.dispatch(
            cpass,
            &ctx.params(step, 0),
            &self.pipelines.zero_kspace,
            bg,
            ew,
            "zero_ksp",
        );
        let source_params = PstdParams {
            axis: injection.count as u32,
            ..ctx.params(step, 0)
        };
        self.dispatch(
            cpass,
            &source_params,
            injection.inject_pipeline,
            injection.bind_group,
            StepCtx::ceil_div(injection.count, 256),
            "inject_source",
        );
        if injection.apply_correction {
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
        }
        self.dispatch(
            cpass,
            &ctx.params(step, 0),
            injection.add_pipeline,
            bg,
            ew,
            "add_source",
        );
    }

    /// Encode the velocity source after the velocity update.
    pub(in crate::pstd_gpu) fn encode_velocity_source_injection(
        &self,
        cpass: &mut wgpu::ComputePass<'_>,
        ctx: &StepCtx,
        bg: &wgpu::BindGroup,
        bg_vel: &wgpu::BindGroup,
        step: u32,
        source_active: bool,
    ) {
        if ctx.n_vel_x > 0 && source_active {
            self.encode_source_injection(
                cpass,
                ctx,
                bg,
                step,
                SourceInjection {
                    bind_group: bg_vel,
                    count: ctx.n_vel_x,
                    apply_correction: ctx.velocity_source_correction,
                    inject_pipeline: &self.pipelines.inject_vel_x,
                    add_pipeline: &self.pipelines.add_kspace_to_field_ux,
                },
            );
        }
    }

    /// Encode the pressure source after the density update.
    pub(in crate::pstd_gpu) fn encode_pressure_source_injection(
        &self,
        cpass: &mut wgpu::ComputePass<'_>,
        ctx: &StepCtx,
        bg: &wgpu::BindGroup,
        step: u32,
        source_active: bool,
    ) {
        if ctx.n_src > 0 && source_active {
            self.encode_source_injection(
                cpass,
                ctx,
                bg,
                step,
                SourceInjection {
                    bind_group: bg,
                    count: ctx.n_src,
                    apply_correction: ctx.pressure_source_correction,
                    inject_pipeline: &self.pipelines.inject_src,
                    add_pipeline: &self.pipelines.add_kspace_to_density,
                },
            );
        }
    }
}
