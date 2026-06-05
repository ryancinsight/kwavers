//! GPU PSTD solver construction tests.

use super::super::{AbsorptionArrays, GpuPstdSolver, MediumArrays, PmlArrays, SolverParams};
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use std::sync::Arc;

/// Verify GpuPstdSolver can be constructed and runs without error.
/// Skipped if no GPU adapter is available (headless CI).
/// # Panics
/// - Panics if `device creation`.
///
#[test]
fn test_gpu_pstd_solver_new() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }));

    let Ok(adapter) = adapter else {
        eprintln!("No GPU adapter — skipping GpuPstdSolver test");
        return;
    };

    let native_limits = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("test_pstd"),
        required_features: wgpu::Features::PUSH_CONSTANTS,
        required_limits: wgpu::Limits {
            max_push_constant_size: 128,
            max_storage_buffers_per_shader_stage:
                native_limits.max_storage_buffers_per_shader_stage,
            ..wgpu::Limits::default()
        },
        memory_hints: wgpu::MemoryHints::default(),
        trace: wgpu::Trace::Off,
    }))
    .expect("device creation");

    let device = Arc::new(device);
    let queue = Arc::new(queue);

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

    let solver = GpuPstdSolver::new(
        device,
        queue,
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
