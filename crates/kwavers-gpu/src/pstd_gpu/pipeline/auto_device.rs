//! `GpuPstdSolver::with_auto_device` — automatic GPU adapter selection.
//!
//! SRP: changes when adapter selection policy or device descriptor changes.

use super::super::{
    state::{PstdAutoDeviceProvider, WgpuPstdStateProvider},
    GpuPstdSolver,
};
use super::{AbsorptionArrays, MediumArrays, PmlArrays, SolverParams};
use crate::{backend::init::GpuProviderContext, gpu::GpuDeviceProvider};
use hephaestus_core::DeviceFeature;
use hephaestus_wgpu::WgpuDevice;
use kwavers_grid::Grid;

impl PstdAutoDeviceProvider for WgpuPstdStateProvider {
    fn acquire_auto_context() -> Result<Self::Context, String> {
        GpuProviderContext::<WgpuDevice>::with_features_and_limits(
            WgpuDevice::acquisition_preference(),
            &[DeviceFeature::PushConstants],
            pstd_required_limits(),
        )
        .map_err(|e| format!("GPU device creation failed: {e}"))
    }
}

impl<P> GpuPstdSolver<P>
where
    P: PstdAutoDeviceProvider,
{
    /// Create a `GpuPstdSolver` by automatically selecting the best available
    /// GPU adapter.  Returns `Err` if no adapter is found or device creation fails.
    ///
    /// This constructor delegates device acquisition to Hephaestus, so PSTD
    /// joins the Atlas GPU provider seam while the existing kernels continue
    /// to consume the raw WGPU handles exposed by the provider.
    pub fn with_auto_device(
        grid: &Grid,
        medium: MediumArrays<'_>,
        solver: SolverParams,
        pml: PmlArrays<'_>,
        absorption: AbsorptionArrays<'_>,
    ) -> Result<Self, String> {
        let context = P::acquire_auto_context()?;

        Self::new(context, grid, medium, solver, pml, absorption)
    }
}

fn pstd_required_limits() -> hephaestus_core::DeviceLimits {
    hephaestus_core::DeviceLimits {
        max_push_constant_size: 128,
        ..WgpuDevice::required_limits()
    }
}
