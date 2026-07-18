//! PSTD pass-body provider contracts.

use super::super::{state::WgpuPstdState, PstdParams};
use super::commands::{PstdCommandProvider, WgpuPstdCommandProvider};
use super::encode::StepCtx;

/// Provider contract for PSTD compute-pass body encoding.
pub(super) trait PstdPassProvider {
    /// Command provider whose compute-pass type this pass provider encodes into.
    type CommandProvider: PstdCommandProvider;

    /// Provider-owned bind group type.
    type BindGroup;

    /// Encode the zero-field pass body.
    fn encode_zero_fields<'pass>(
        &self,
        cpass: &mut <Self::CommandProvider as PstdCommandProvider>::ComputePass<'pass>,
        params: &PstdParams,
        sensor_bind_group: &Self::BindGroup,
        elem_workgroups: u32,
    );

    /// Encode one PSTD time-step pass body.
    fn encode_time_step<'pass>(
        &self,
        cpass: &mut <Self::CommandProvider as PstdCommandProvider>::ComputePass<'pass>,
        ctx: &StepCtx,
        sensor_bind_group: &Self::BindGroup,
        velocity_sensor_bind_group: &Self::BindGroup,
        step: u32,
        pressure_source_active: bool,
        velocity_source_active: bool,
    );
}

/// WGPU PSTD pass-body provider.
pub(super) struct WgpuPstdPassProvider<'solver> {
    state: &'solver WgpuPstdState,
}

impl<'solver> WgpuPstdPassProvider<'solver> {
    /// Create a WGPU pass-body provider for PSTD state.
    #[must_use]
    pub(super) const fn new(state: &'solver WgpuPstdState) -> Self {
        Self { state }
    }
}

impl<'solver> PstdPassProvider for WgpuPstdPassProvider<'solver> {
    type CommandProvider = WgpuPstdCommandProvider<'solver>;
    type BindGroup = wgpu::BindGroup;

    fn encode_zero_fields<'pass>(
        &self,
        cpass: &mut <Self::CommandProvider as PstdCommandProvider>::ComputePass<'pass>,
        params: &PstdParams,
        sensor_bind_group: &Self::BindGroup,
        elem_workgroups: u32,
    ) {
        self.state.dispatch(
            cpass,
            params,
            &self.state.pipelines.zero_fields,
            sensor_bind_group,
            elem_workgroups,
            "zero_fields",
        );
    }

    fn encode_time_step<'pass>(
        &self,
        cpass: &mut <Self::CommandProvider as PstdCommandProvider>::ComputePass<'pass>,
        ctx: &StepCtx,
        sensor_bind_group: &Self::BindGroup,
        velocity_sensor_bind_group: &Self::BindGroup,
        step: u32,
        pressure_source_active: bool,
        velocity_source_active: bool,
    ) {
        self.state
            .encode_velocity_update(cpass, ctx, sensor_bind_group, step);
        self.state.encode_velocity_source_injection(
            cpass,
            ctx,
            sensor_bind_group,
            velocity_sensor_bind_group,
            step,
            velocity_source_active,
        );
        self.state
            .encode_nonlinear_snapshot(cpass, ctx, sensor_bind_group, step);
        self.state
            .encode_density_update(cpass, ctx, sensor_bind_group, step);
        self.state.encode_pressure_source_injection(
            cpass,
            ctx,
            sensor_bind_group,
            step,
            pressure_source_active,
        );
        self.state
            .encode_pressure_record(cpass, ctx, sensor_bind_group, step);
    }
}

#[cfg(test)]
mod tests {
    use super::{PstdPassProvider, WgpuPstdPassProvider};

    #[test]
    fn pstd_pass_provider_is_generic_over_provider_trait() {
        fn assert_provider<P>()
        where
            P: PstdPassProvider + 'static,
            P::CommandProvider: 'static,
        {
            let _ = core::mem::size_of::<P::BindGroup>();
            let _ = core::mem::size_of::<
                <P::CommandProvider as super::PstdCommandProvider>::ComputePass<'static>,
            >();
        }

        assert_provider::<WgpuPstdPassProvider<'static>>();
    }
}
