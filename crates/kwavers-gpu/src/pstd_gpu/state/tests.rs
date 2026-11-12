//! Tests of the provider state contracts.

use crate::backend::init::GpuProviderContext;
use hephaestus_wgpu::WgpuDevice;

use super::{
    PstdAutoDeviceProvider, PstdMediumUpdateState, PstdOutputRequest, PstdRunInputs,
    PstdRunScalars, PstdRunState, PstdStateBuilder, PstdStateProvider, WgpuPstdState,
    WgpuPstdStateProvider,
};

#[test]
fn pstd_output_request_selects_final_and_peak_independently() {
    let traces = PstdOutputRequest::sensor_traces();
    let final_fields = PstdOutputRequest::with_final_fields();
    let peak_pressure = PstdOutputRequest::with_peak_pressure();
    let both = PstdOutputRequest::with_final_fields_and_peak_pressure();

    assert_eq!(
        (
            traces.includes_final_fields(),
            traces.includes_peak_pressure(),
        ),
        (false, false)
    );
    assert_eq!(
        (
            final_fields.includes_final_fields(),
            final_fields.includes_peak_pressure(),
        ),
        (true, false)
    );
    assert_eq!(
        (
            peak_pressure.includes_final_fields(),
            peak_pressure.includes_peak_pressure(),
        ),
        (false, true)
    );
    assert_eq!(
        (both.includes_final_fields(), both.includes_peak_pressure()),
        (true, true)
    );
}

#[test]
fn pstd_solver_state_is_provider_associated() {
    fn assert_state_provider<P>()
    where
        P: PstdStateProvider<State = WgpuPstdState>,
    {
        let _ = core::mem::size_of::<P::State>();
    }

    assert_state_provider::<WgpuPstdStateProvider>();
}

#[test]
fn pstd_solver_state_builder_uses_provider_handles() {
    fn assert_state_builder<P>()
    where
        P: PstdStateBuilder<State = WgpuPstdState, Context = GpuProviderContext<WgpuDevice>>,
    {
        let _ = core::mem::size_of::<P::Context>();
    }

    assert_state_builder::<WgpuPstdStateProvider>();
}

#[test]
fn pstd_solver_auto_device_provider_uses_provider_handles() {
    fn assert_auto_device_provider<P>()
    where
        P: PstdAutoDeviceProvider<State = WgpuPstdState, Context = GpuProviderContext<WgpuDevice>>,
    {
        let _ = core::mem::size_of::<P::Context>();
    }

    assert_auto_device_provider::<WgpuPstdStateProvider>();
}

#[test]
fn pstd_solver_run_state_is_provider_owned() {
    fn assert_run_state<S>()
    where
        S: PstdRunState,
    {
        let _ = core::mem::size_of::<S>();
        let _ = core::mem::size_of::<PstdRunScalars>();
        let _ = core::mem::size_of::<PstdRunInputs<'static>>();
    }

    assert_run_state::<WgpuPstdState>();
}

#[test]
fn pstd_solver_medium_update_state_is_provider_owned() {
    fn assert_medium_update_state<S>()
    where
        S: PstdMediumUpdateState,
    {
        let _ = core::mem::size_of::<S>();
    }

    assert_medium_update_state::<WgpuPstdState>();
}
