//! `GpuPstdSolver::with_auto_device` — automatic GPU adapter selection.
//!
//! SRP: changes when adapter selection policy or device descriptor changes.

use super::super::{
    state::{
        PstdAutoDeviceProvider, WgpuPstdStateProvider,
        ABSORPTION_PIPELINE_BUFFERS_PER_SHADER_STAGE, LOSSLESS_PIPELINE_BUFFERS_PER_SHADER_STAGE,
    },
    GpuPstdSolver,
};
use super::{AbsorptionArrays, MediumArrays, PmlArrays, SolverParams};
use crate::{backend::init::GpuProviderContext, gpu::GpuDeviceProvider};
use hephaestus_core::DeviceFeature;
use hephaestus_wgpu::WgpuDevice;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;

impl PstdAutoDeviceProvider for WgpuPstdStateProvider {
    fn acquire_auto_context(absorbing: bool) -> KwaversResult<Self::Context> {
        GpuProviderContext::<WgpuDevice>::with_features_and_limits(
            WgpuDevice::acquisition_preference(),
            &[DeviceFeature::ImmediateData],
            pstd_required_limits(absorbing),
        )
    }
}

impl<P> GpuPstdSolver<P>
where
    P: PstdAutoDeviceProvider,
{
    /// Create a `GpuPstdSolver` by automatically selecting the best available
    /// GPU adapter.
    ///
    /// This constructor delegates device acquisition to Hephaestus, so PSTD
    /// joins the Atlas GPU provider seam while the existing kernels continue
    /// to consume the raw WGPU handles exposed by the provider.
    ///
    /// # Errors
    ///
    /// - `SystemError::GpuNotAvailable` when this host has no compatible
    ///   accelerator adapter (typed, so a caller can select a CPU backend);
    /// - `KwaversError::GpuError` when a present adapter fails device creation;
    /// - `KwaversError::InvalidInput` when the solver rejects its parameters or
    ///   fails to build its device state.
    pub fn with_auto_device(
        grid: &Grid,
        medium: MediumArrays<'_>,
        solver: SolverParams,
        pml: PmlArrays<'_>,
        absorption: AbsorptionArrays<'_>,
    ) -> KwaversResult<Self> {
        let context = P::acquire_auto_context(solver.absorbing)?;

        Self::new(context, grid, medium, solver, pml, absorption).map_err(|error| {
            KwaversError::InvalidInput(format!("GPU PSTD solver construction failed: {error}"))
        })
    }
}

fn pstd_required_limits(absorbing: bool) -> hephaestus_core::DeviceLimits {
    hephaestus_core::DeviceLimits {
        max_storage_buffers_per_shader_stage: Some(if absorbing {
            ABSORPTION_PIPELINE_BUFFERS_PER_SHADER_STAGE
        } else {
            LOSSLESS_PIPELINE_BUFFERS_PER_SHADER_STAGE
        }),
        max_immediate_size: 128,
        ..WgpuDevice::required_limits()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        pstd_required_limits, ABSORPTION_PIPELINE_BUFFERS_PER_SHADER_STAGE,
        LOSSLESS_PIPELINE_BUFFERS_PER_SHADER_STAGE,
    };

    #[test]
    fn pstd_device_limits_match_the_enabled_pipeline_bindings() {
        assert_eq!(
            pstd_required_limits(false).max_storage_buffers_per_shader_stage,
            Some(LOSSLESS_PIPELINE_BUFFERS_PER_SHADER_STAGE)
        );
        assert_eq!(
            pstd_required_limits(true).max_storage_buffers_per_shader_stage,
            Some(ABSORPTION_PIPELINE_BUFFERS_PER_SHADER_STAGE)
        );
        assert!(pstd_required_limits(false).max_immediate_size >= 48);
    }
}
