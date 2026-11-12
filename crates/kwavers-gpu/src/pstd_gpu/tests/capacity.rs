//! Device tests of the plan capacity check, on devices acquired with a
//! lowered `max_buffer_size`.

use super::super::{
    AbsorptionArrays, GpuPstdSolver, MediumArrays, PmlArrays, SolverParams, WgpuPstdStateProvider,
};
use crate::pstd_gpu::state::LOSSLESS_PIPELINE_BUFFERS_PER_SHADER_STAGE;
use crate::{backend::init::GpuProviderContext, gpu::GpuDeviceProvider};
use hephaestus_core::{DeviceFeature, DeviceLimits};
use hephaestus_wgpu::WgpuDevice;
use kwavers_grid::Grid;

const SHAPE: [usize; 3] = [16, 16, 16];
/// One field buffer of the grid: one `f32` value for each of 4096 points.
const FIELD_BYTES: u64 = 4096 * 4;
/// `pml_xyz` of the grid: three `f32` values for each of 4096 points.
const PACKED_PML_BYTES: u64 = 3 * FIELD_BYTES;

/// A lossless `SHAPE` solver on a device with the given `max_buffer_size`.
fn solver_at_limit(max_buffer_size: u64) -> Result<GpuPstdSolver, String> {
    let context = GpuProviderContext::<WgpuDevice>::with_features_and_limits(
        WgpuDevice::acquisition_preference(),
        &[DeviceFeature::ImmediateData],
        DeviceLimits {
            max_buffer_size,
            max_storage_buffers_per_shader_stage: Some(LOSSLESS_PIPELINE_BUFFERS_PER_SHADER_STAGE),
            max_immediate_size: 128,
            ..WgpuDevice::required_limits()
        },
    )
    .unwrap_or_else(|error| panic!("an ignored device test needs an adapter: {error:?}"));
    let [nx, ny, nz] = SHAPE;
    let points = nx * ny * nz;
    let sound_speed = vec![1500.0f32; points];
    let density = vec![1000.0f32; points];
    let ones = vec![1.0f32; points];
    let zeros = vec![0.0f32; points];
    GpuPstdSolver::<WgpuPstdStateProvider>::new(
        context,
        &Grid::new(nx, ny, nz, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid test grid"),
        MediumArrays {
            c0_flat: &sound_speed,
            rho0_flat: &density,
        },
        SolverParams {
            dt: 1.0e-8,
            nt: 4,
            c_ref: 1500.0,
            nonlinear: false,
            absorbing: false,
        },
        PmlArrays {
            x: &ones,
            y: &ones,
            z: &ones,
            sgx: &ones,
            sgy: &ones,
            sgz: &ones,
        },
        AbsorptionArrays {
            bon_a_flat: &zeros,
            nabla1: &zeros,
            nabla2: &zeros,
            tau: &zeros,
            eta: &zeros,
        },
    )
}

/// Ties the planned byte count to the allocator: at a limit equal to the
/// packed PML buffer the device creates every buffer of the plan.
#[test]
#[ignore = "requires a GPU device"]
fn a_plan_at_the_buffer_limit_builds() {
    let solver = solver_at_limit(PACKED_PML_BYTES)
        .unwrap_or_else(|message| panic!("the plan fits the device limit: {message}"));

    assert_eq!([solver.nx, solver.ny, solver.nz], SHAPE);
}

/// The seven field buffers fit this limit and the packed PML buffer does not,
/// so a check that counts one value per grid point admits the plan.
#[test]
#[ignore = "requires a GPU device"]
fn a_limit_between_a_field_and_the_packed_pml_is_refused() {
    assert_eq!(
        solver_at_limit(2 * FIELD_BYTES).expect_err("the plan exceeds the device limit"),
        "Resource limit exceeded: GPU PSTD buffer `pml_xyz` needs 49152 bytes; the device enables max_buffer_size = 32768 bytes"
    );
}

/// The first field buffer already exceeds this limit, so a check that runs
/// after the provider creates it ends in the `wgpu` validation panic.
#[test]
#[ignore = "requires a GPU device"]
fn a_limit_below_one_field_is_refused_before_any_buffer() {
    assert_eq!(
        solver_at_limit(FIELD_BYTES / 2).expect_err("the plan exceeds the device limit"),
        "Resource limit exceeded: GPU PSTD buffer `pml_xyz` needs 49152 bytes; the device enables max_buffer_size = 8192 bytes"
    );
}
