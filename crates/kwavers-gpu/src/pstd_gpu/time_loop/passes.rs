//! PSTD pass-body provider contracts.

use super::super::{state::WgpuPstdState, PstdParams};
use super::encode::StepCtx;
use hephaestus_core::Result;
use hephaestus_wgpu::WgpuGroupedSequence;

/// Bind groups required by one monomorphic PSTD time-step pass.
pub(super) struct StepBindGroups<'a, B> {
    pub(super) sensor: &'a B,
    pub(super) velocity_sensor: &'a B,
}

/// Source-activity flags derived once for one PSTD time step.
pub(super) struct SourceActivity {
    pub(super) pressure: bool,
    pub(super) velocity: bool,
}

/// Provider contract for PSTD compute-pass body encoding.
pub(super) trait PstdPassProvider {
    /// Provider-owned bind group type.
    type BindGroup;

    /// Encode the zero-field pass body.
    fn encode_zero_fields<'pass>(
        &self,
        sequence: &mut WgpuGroupedSequence<'pass>,
        params: &PstdParams,
        sensor_bind_group: &Self::BindGroup,
        elem_workgroups: u32,
    );

    /// Encode one PSTD time-step pass body.
    fn encode_time_step<'pass>(
        &self,
        sequence: &mut WgpuGroupedSequence<'pass>,
        ctx: &StepCtx,
        bind_groups: StepBindGroups<'_, Self::BindGroup>,
        step: u32,
        source_activity: SourceActivity,
    ) -> Result<()>;
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
    type BindGroup = wgpu::BindGroup;

    fn encode_zero_fields<'pass>(
        &self,
        sequence: &mut WgpuGroupedSequence<'pass>,
        params: &PstdParams,
        sensor_bind_group: &Self::BindGroup,
        elem_workgroups: u32,
    ) {
        self.state.dispatch(
            sequence,
            params,
            &self.state.pipelines.zero_fields,
            sensor_bind_group,
            elem_workgroups,
            "zero_fields",
        );
    }

    fn encode_time_step<'pass>(
        &self,
        sequence: &mut WgpuGroupedSequence<'pass>,
        ctx: &StepCtx,
        bind_groups: StepBindGroups<'_, Self::BindGroup>,
        step: u32,
        source_activity: SourceActivity,
    ) -> Result<()> {
        self.state
            .encode_velocity_update(sequence, ctx, bind_groups.sensor, step)?;
        self.state.encode_velocity_source_injection(
            sequence,
            ctx,
            bind_groups.sensor,
            bind_groups.velocity_sensor,
            step,
            source_activity.velocity,
        )?;
        self.state
            .encode_nonlinear_snapshot(sequence, ctx, bind_groups.sensor, step);
        self.state
            .encode_density_update(sequence, ctx, bind_groups.sensor, step)?;
        self.state.encode_pressure_source_injection(
            sequence,
            ctx,
            bind_groups.sensor,
            step,
            source_activity.pressure,
        )?;
        self.state
            .encode_pressure_record(sequence, ctx, bind_groups.sensor, step);
        Ok(())
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
        {
            let _ = core::mem::size_of::<P::BindGroup>();
        }

        assert_provider::<WgpuPstdPassProvider<'static>>();
    }
}
