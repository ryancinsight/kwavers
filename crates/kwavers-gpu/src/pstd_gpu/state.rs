//! Provider-owned PSTD GPU state groups.

use super::pipeline::{AbsorptionArrays, MediumArrays, PmlArrays, SolverParams};
use crate::backend::init::GpuProviderContext;
use hephaestus_wgpu::WgpuDevice;
use kwavers_grid::Grid;

/// Storage-buffer bindings used by the three lossless PSTD bind groups.
pub(super) const LOSSLESS_PIPELINE_BUFFERS_PER_SHADER_STAGE: u32 = 24;
/// Storage-buffer bindings used when fractional-Laplacian absorption is enabled.
pub(super) const ABSORPTION_PIPELINE_BUFFERS_PER_SHADER_STAGE: u32 = 32;

/// Provider contract for PSTD solver state ownership.
pub trait PstdStateProvider {
    /// Concrete state object owned by a PSTD solver for this provider.
    type State;
}

/// Provider contract for PSTD solver state construction.
pub trait PstdStateBuilder: PstdStateProvider {
    /// Provider-owned execution context consumed by PSTD construction.
    type Context;

    /// Build provider-owned PSTD state.
    fn build_state(
        context: Self::Context,
        grid: &Grid,
        medium: MediumArrays<'_>,
        solver: SolverParams,
        pml: PmlArrays<'_>,
        absorption: AbsorptionArrays<'_>,
    ) -> Result<Self::State, String>;
}

/// Provider contract for automatic PSTD GPU device acquisition.
pub trait PstdAutoDeviceProvider: PstdStateBuilder {
    /// Acquire a provider-owned execution context for PSTD execution.
    fn acquire_auto_context(absorbing: bool) -> Result<Self::Context, String>;
}

/// Scalar run metadata supplied by the public PSTD solver wrapper.
#[derive(Clone, Copy)]
pub struct PstdRunScalars {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub nt: usize,
    pub dt: f64,
    pub nonlinear: bool,
    pub absorbing: bool,
}

/// Host outputs requested from one GPU PSTD run.
///
/// Sensor traces are always available. Final fields and the pointwise temporal
/// pressure envelope are independent because a final frame is not generally a
/// peak-pressure field for a transient burst.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PstdOutputRequest {
    final_fields: bool,
    peak_pressure: bool,
}

impl PstdOutputRequest {
    /// Keep field state on the provider and download only sensor traces.
    #[must_use]
    pub const fn sensor_traces() -> Self {
        Self {
            final_fields: false,
            peak_pressure: false,
        }
    }

    /// Download sensor traces and final pressure plus staggered velocity fields.
    #[must_use]
    pub const fn with_final_fields() -> Self {
        Self {
            final_fields: true,
            peak_pressure: false,
        }
    }

    /// Download sensor traces and the pointwise `max_t |p|` pressure envelope.
    #[must_use]
    pub const fn with_peak_pressure() -> Self {
        Self {
            final_fields: false,
            peak_pressure: true,
        }
    }

    /// Download sensor traces, final fields, and the peak-pressure envelope.
    #[must_use]
    pub const fn with_final_fields_and_peak_pressure() -> Self {
        Self {
            final_fields: true,
            peak_pressure: true,
        }
    }

    pub(super) const fn includes_final_fields(self) -> bool {
        self.final_fields
    }

    pub(super) const fn includes_peak_pressure(self) -> bool {
        self.peak_pressure
    }
}

/// Borrowed PSTD run inputs supplied by the public solver wrapper.
pub struct PstdRunInputs<'a> {
    pub sensor_indices: &'a [u32],
    pub source_indices: &'a [u32],
    pub source_signals: &'a [f32],
    /// Whether the pressure source requires k-space correction.
    pub pressure_source_correction: bool,
    pub vel_x_indices: &'a [u32],
    pub vel_x_signals: &'a [f32],
    /// Whether the x-velocity source requires k-space correction.
    pub velocity_source_correction: bool,
    pub output_request: PstdOutputRequest,
}

/// Provider-read final acoustic state in row-major grid order.
#[derive(Debug)]
pub struct PstdFinalFields {
    /// Final acoustic pressure field.
    pub pressure: Vec<f32>,
    /// Final staggered x-velocity field.
    pub velocity_x: Vec<f32>,
    /// Final staggered y-velocity field.
    pub velocity_y: Vec<f32>,
    /// Final staggered z-velocity field.
    pub velocity_z: Vec<f32>,
}

/// Host result from one GPU PSTD batch.
#[derive(Debug)]
pub struct PstdRunResult {
    /// Pressure traces in `[sensor * time_steps + step]` order.
    pub sensor_data: Vec<f32>,
    /// Final fields when requested by [`PstdOutputRequest`].
    pub final_fields: Option<PstdFinalFields>,
    /// Pointwise `max_t |p|` in row-major grid order when requested.
    pub peak_pressure: Option<Vec<f32>>,
}

/// Provider-state contract for executing a PSTD run.
pub trait PstdRunState {
    /// Execute a PSTD run with provider-owned state.
    fn run_pstd(&mut self, scalars: PstdRunScalars, inputs: PstdRunInputs<'_>) -> PstdRunResult;
}

/// Provider-state contract for updating PSTD medium-dependent buffers.
pub trait PstdMediumUpdateState {
    /// Update sound-speed and density dependent buffers.
    fn update_medium_variable<T>(&mut self, c0_flat: &[T], rho0_flat: &[T])
    where
        T: Into<f64> + Copy;

    /// Update all medium, nonlinearity, and absorption buffers.
    fn update_medium<T>(
        &mut self,
        c0_flat: &[T],
        rho0_flat: &[T],
        bon_a_flat: &[f32],
        absorb_tau_flat: &[f32],
        absorb_eta_flat: &[f32],
    ) where
        T: Into<f64> + Copy;

    /// Disable source k-space correction by overwriting the source-kappa buffer.
    fn disable_source_correction(&self);
}

/// WGPU PSTD state provider.
pub struct WgpuPstdStateProvider;

/// WGPU-owned PSTD acoustic field buffers.
pub(super) struct WgpuPstdFieldBuffers {
    pub(super) p: wgpu::Buffer,
    pub(super) ux: wgpu::Buffer,
    pub(super) uy: wgpu::Buffer,
    pub(super) uz: wgpu::Buffer,
    pub(super) rhox: wgpu::Buffer,
    pub(super) rhoy: wgpu::Buffer,
    pub(super) rhoz: wgpu::Buffer,
}

/// WGPU-owned PSTD k-space, medium, and source-correction buffers.
pub(super) struct WgpuPstdMediumBuffers {
    pub(super) kappa: wgpu::Buffer,
    pub(super) rho0_inv: wgpu::Buffer,
    pub(super) c0_sq: wgpu::Buffer,
    pub(super) rho0: wgpu::Buffer,
    /// B/(2A) per voxel; 0.0 for linear simulations.
    pub(super) bon_a: wgpu::Buffer,
    /// FFT twiddle table for the current PSTD shader.
    pub(super) twiddle_fft: wgpu::Buffer,
    pub(super) source_kappa: wgpu::Buffer,
}

/// WGPU-owned PSTD k-space work buffers.
pub(super) struct WgpuPstdKspaceBuffers {
    pub(super) re: wgpu::Buffer,
    pub(super) im: wgpu::Buffer,
}

/// WGPU-owned PSTD fractional-Laplacian absorption buffers.
pub(super) struct WgpuPstdAbsorptionBuffers {
    /// |k|^(y-2) in FFT order.
    pub(super) nabla1: wgpu::Buffer,
    /// |k|^(y-1) in FFT order.
    pub(super) nabla2: wgpu::Buffer,
    /// -2*alpha0*c0^(y-1) per voxel.
    pub(super) tau: wgpu::Buffer,
    /// 2*alpha0*c0^y*tan(pi*y/2) per voxel.
    pub(super) eta: wgpu::Buffer,
    pub(super) scratch_kre: wgpu::Buffer,
    pub(super) scratch_kim: wgpu::Buffer,
    pub(super) scratch_l1: wgpu::Buffer,
    pub(super) scratch_l2: wgpu::Buffer,
}

/// WGPU-owned PSTD PML and packed k-space shift buffers.
pub(super) struct WgpuPstdPmlShiftBuffers {
    pub(super) pml_sgx: wgpu::Buffer,
    pub(super) pml_sgy: wgpu::Buffer,
    pub(super) pml_sgz: wgpu::Buffer,
    /// Packed `[pml_x | pml_y | pml_z]`.
    pub(super) pml_xyz: wgpu::Buffer,
    /// Packed 12 x 1D shift arrays.
    pub(super) shifts_all: wgpu::Buffer,
}

/// WGPU-owned PSTD run-cache state rebuilt when sensor/source layout changes.
#[derive(Default)]
pub(super) struct WgpuPstdRunCache {
    pub(super) sensor_indices_buf: Option<wgpu::Buffer>,
    pub(super) sensor_data_buf: Option<wgpu::Buffer>,
    pub(super) source_data_buf: Option<wgpu::Buffer>,
    pub(super) vel_x_data_buf: Option<wgpu::Buffer>,
    pub(super) staging_buf: Option<wgpu::Buffer>,
    pub(super) field_staging_buf: Option<wgpu::Buffer>,
    pub(super) bg_sensor: Option<wgpu::BindGroup>,
    pub(super) bg_sensor_vel: Option<wgpu::BindGroup>,
    pub(super) output_storage_len: usize,
    pub(super) peak_offset: usize,
    pub(super) records_peak_pressure: bool,
    pub(super) n_sensors: usize,
    pub(super) n_src: usize,
    pub(super) n_vel_x: usize,
}

/// WGPU-owned PSTD permanent bind groups reused across runs.
pub(super) struct WgpuPstdPermanentBindGroups {
    pub(super) fields: wgpu::BindGroup,
    pub(super) kspace: wgpu::BindGroup,
    /// Absorption operator bind group.
    pub(super) absorb: Option<wgpu::BindGroup>,
}

/// WGPU-owned PSTD layout handles retained after construction.
pub(super) struct WgpuPstdLayouts {
    pub(super) sensor: wgpu::BindGroupLayout,
}

/// WGPU-owned fractional-Laplacian absorption pipelines.
pub(super) struct WgpuPstdAbsorptionPipelines {
    pub(super) mul_nabla: wgpu::ComputePipeline,
    pub(super) copy_to_scratch: wgpu::ComputePipeline,
    pub(super) accum_div_u: wgpu::ComputePipeline,
    pub(super) prep_l1_kspace: wgpu::ComputePipeline,
    pub(super) prep_l2_kspace: wgpu::ComputePipeline,
    pub(super) pressure_correction: wgpu::ComputePipeline,
    pub(super) save_kspace: wgpu::ComputePipeline,
    pub(super) restore_and_shift: wgpu::ComputePipeline,
}

/// WGPU-owned PSTD compute pipelines.
pub(super) struct WgpuPstdPipelines {
    pub(super) zero_fields: wgpu::ComputePipeline,
    pub(super) zero_kspace: wgpu::ComputePipeline,
    pub(super) fft: wgpu::ComputePipeline,
    pub(super) kspace_shift: wgpu::ComputePipeline,
    pub(super) vel_update: wgpu::ComputePipeline,
    pub(super) dens_update: wgpu::ComputePipeline,
    pub(super) snapshot_rho0_plus_rho: wgpu::ComputePipeline,
    pub(super) pres_density: wgpu::ComputePipeline,
    pub(super) record: wgpu::ComputePipeline,
    pub(super) peak_pressure: wgpu::ComputePipeline,
    pub(super) inject_src: wgpu::ComputePipeline,
    pub(super) inject_vel_x: wgpu::ComputePipeline,
    pub(super) apply_source_kappa: wgpu::ComputePipeline,
    pub(super) add_kspace_to_field_ux: wgpu::ComputePipeline,
    pub(super) add_kspace_to_density: wgpu::ComputePipeline,
    pub(super) copy_field_to_k: wgpu::ComputePipeline,
    /// Present only when the solver enables fractional-Laplacian absorption.
    pub(super) absorption: Option<WgpuPstdAbsorptionPipelines>,
}

/// WGPU-owned PSTD solver state.
///
/// Some buffers are only read by permanent bind groups after construction; the
/// state object must still own them so those bindings remain valid.
#[allow(dead_code)]
pub struct WgpuPstdState {
    pub(super) context: GpuProviderContext<WgpuDevice>,
    pub(super) field_buffers: WgpuPstdFieldBuffers,
    pub(super) kspace_buffers: WgpuPstdKspaceBuffers,
    pub(super) medium_buffers: WgpuPstdMediumBuffers,
    pub(super) absorption_buffers: WgpuPstdAbsorptionBuffers,
    pub(super) pml_shift_buffers: WgpuPstdPmlShiftBuffers,
    pub(super) pipelines: WgpuPstdPipelines,
    pub(super) permanent_bind_groups: WgpuPstdPermanentBindGroups,
    pub(super) layouts: WgpuPstdLayouts,
    pub(super) run_cache: WgpuPstdRunCache,
    /// Preallocated host buffer for medium sound-speed-squared uploads.
    pub(super) scratch_c0_sq: Vec<f32>,
    /// Preallocated host buffer for inverse-density uploads.
    pub(super) scratch_rho0_inv: Vec<f32>,
    /// Preallocated host buffer for density uploads.
    pub(super) scratch_rho0_flat: Vec<f32>,
    /// Persistent unity upload buffer for source-correction disablement.
    pub(super) scratch_source_kappa_ones: Vec<f32>,
    /// Packed host-side pressure source upload buffer.
    pub(super) scratch_source_data: Vec<f32>,
    /// Packed host-side x-velocity source upload buffer.
    pub(super) scratch_vel_x_data: Vec<f32>,
}

impl PstdStateProvider for WgpuPstdStateProvider {
    type State = WgpuPstdState;
}

impl WgpuPstdState {
    pub(in crate::pstd_gpu) fn device(&self) -> &wgpu::Device {
        self.context.device()
    }

    pub(in crate::pstd_gpu) fn queue(&self) -> &wgpu::Queue {
        self.context.queue()
    }

    pub(super) fn absorption_pipelines(&self) -> &WgpuPstdAbsorptionPipelines {
        self.pipelines
            .absorption
            .as_ref()
            .expect("invariant: absorption dispatch requires absorption-enabled PSTD construction")
    }
}

#[cfg(test)]
mod tests {
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
            P: PstdAutoDeviceProvider<
                State = WgpuPstdState,
                Context = GpuProviderContext<WgpuDevice>,
            >,
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
}
