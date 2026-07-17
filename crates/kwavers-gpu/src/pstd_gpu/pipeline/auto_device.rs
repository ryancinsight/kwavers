//! `GpuPstdSolver::with_auto_device` — automatic GPU adapter selection.
//!
//! SRP: changes when adapter selection policy or device descriptor changes.

use super::super::{
    state::{
        PstdAutoDeviceProvider, WgpuPstdStateProvider,
        ABSORPTION_PIPELINE_BUFFERS_PER_SHADER_STAGE, LOSSLESS_PIPELINE_BUFFERS_PER_SHADER_STAGE,
    },
    GpuPstdSolver, GPU_PSTD_FFT_WORKGROUP_STORAGE_BYTES,
};
use super::{AbsorptionArrays, MediumArrays, PmlArrays, SolverParams};
use crate::{backend::init::GpuProviderContext, gpu::GpuDeviceProvider};
use hephaestus_core::DeviceFeature;
use hephaestus_wgpu::WgpuDevice;
use kwavers_grid::Grid;

impl PstdAutoDeviceProvider for WgpuPstdStateProvider {
    fn acquire_auto_context(absorbing: bool) -> Result<Self::Context, String> {
        GpuProviderContext::<WgpuDevice>::with_features_and_limits(
            WgpuDevice::acquisition_preference(),
            &[DeviceFeature::ImmediateData],
            pstd_required_limits(absorbing),
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
        let context = P::acquire_auto_context(solver.absorbing)?;

        Self::new(context, grid, medium, solver, pml, absorption)
    }
}

fn pstd_required_limits(absorbing: bool) -> hephaestus_core::DeviceLimits {
    hephaestus_core::DeviceLimits {
        max_storage_buffers_per_shader_stage: Some(if absorbing {
            ABSORPTION_PIPELINE_BUFFERS_PER_SHADER_STAGE
        } else {
            LOSSLESS_PIPELINE_BUFFERS_PER_SHADER_STAGE
        }),
        max_compute_workgroup_storage_size: GPU_PSTD_FFT_WORKGROUP_STORAGE_BYTES,
        max_immediate_size: 128,
        ..WgpuDevice::required_limits()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        pstd_required_limits, ABSORPTION_PIPELINE_BUFFERS_PER_SHADER_STAGE,
        GPU_PSTD_FFT_WORKGROUP_STORAGE_BYTES, LOSSLESS_PIPELINE_BUFFERS_PER_SHADER_STAGE,
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
        assert_eq!(
            pstd_required_limits(false).max_compute_workgroup_storage_size,
            GPU_PSTD_FFT_WORKGROUP_STORAGE_BYTES
        );
    }
}
