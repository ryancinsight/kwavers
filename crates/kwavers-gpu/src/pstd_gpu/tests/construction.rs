//! GPU PSTD solver construction tests.

use super::super::pipeline::{
    PstdBindGroupLayoutProvider, PstdBindGroupProvider, PstdBufferProvider, PstdPipelineProvider,
    WgpuPstdBindGroupFactory, WgpuPstdBindGroupLayoutFactory, WgpuPstdBufferFactory,
    WgpuPstdPipelineFactory,
};
use super::super::{
    AbsorptionArrays, GpuPstdSolver, MediumArrays, PmlArrays, SolverParams, WgpuPstdStateProvider,
};
use super::helpers::pstd_test_provider;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};

#[test]
fn pstd_buffer_factory_is_generic_over_provider_trait() {
    fn assert_provider<P>()
    where
        P: PstdBufferProvider,
    {
        let _ = core::mem::size_of::<P::Buffer>();
    }

    assert_provider::<WgpuPstdBufferFactory<'static>>();
}

#[test]
fn pstd_pipeline_factory_is_generic_over_provider_trait() {
    fn assert_provider<P>()
    where
        P: PstdPipelineProvider,
    {
        let _ = core::mem::size_of::<P::Pipeline>();
    }

    assert_provider::<WgpuPstdPipelineFactory<'static>>();
}

#[test]
fn pstd_bind_group_layout_factory_is_generic_over_provider_trait() {
    fn assert_provider<P>()
    where
        P: PstdBindGroupLayoutProvider,
    {
        let _ = core::mem::size_of::<P::BindGroupLayout>();
    }

    assert_provider::<WgpuPstdBindGroupLayoutFactory<'static>>();
}

#[test]
fn pstd_bind_group_factory_is_generic_over_provider_trait() {
    fn assert_provider<P>()
    where
        P: PstdBindGroupProvider,
    {
        let _ = core::mem::size_of::<P::BindGroup>();
    }

    assert_provider::<WgpuPstdBindGroupFactory<'static>>();
}

/// Verify GpuPstdSolver can be constructed and runs without error.
/// Skipped if no GPU adapter is available (headless CI).
#[test]
fn test_gpu_pstd_solver_new() {
    let Some(provider) = pstd_test_provider("test_pstd") else {
        eprintln!("No GPU adapter - skipping GpuPstdSolver test");
        return;
    };

    let n = 32usize;
    let dx = 1e-3_f64;
    let c0 = SOUND_SPEED_WATER_SIM;
    let rho0 = DENSITY_WATER_NOMINAL;
    let dt = 0.3 * dx / c0;
    let nt = 10;

    let grid = kwavers_grid::Grid::new(n, n, n, dx, dx, dx).unwrap();
    let c0v: Vec<f32> = vec![c0 as f32; n * n * n];
    let rho0v: Vec<f32> = vec![rho0 as f32; n * n * n];
    let ones: Vec<f32> = vec![1.0f32; n * n * n];
    let zeros: Vec<f32> = vec![0.0f32; n * n * n];

    let solver = GpuPstdSolver::<WgpuPstdStateProvider>::new(
        provider,
        &grid,
        MediumArrays {
            c0_flat: &c0v,
            rho0_flat: &rho0v,
        },
        SolverParams {
            dt,
            nt,
            c_ref: c0,
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
    );

    assert!(
        solver.is_ok(),
        "GpuPstdSolver::new failed: {:?}",
        solver.err()
    );
    eprintln!("GpuPstdSolver constructed successfully");
}
