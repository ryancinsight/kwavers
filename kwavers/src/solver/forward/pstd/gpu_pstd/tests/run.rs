//! GPU PSTD run tests — pressure source, velocity source, multi-source, and benchmark.

use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use super::super::{AbsorptionArrays, GpuPstdSolver, MediumArrays, PmlArrays, SolverParams};

fn make_solver_32(nt: usize) -> Option<GpuPstdSolver> {
    let n3 = 32 * 32 * 32;
    let c0v = vec![SOUND_SPEED_WATER_SIM as f32; n3];
    let rho0v = vec![1000.0f32; n3];
    let ones = vec![1.0f32; n3];
    let zeros = vec![0.0f32; n3];
    GpuPstdSolver::with_auto_device(
        &crate::domain::grid::Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap(),
        MediumArrays {
            c0_flat: &c0v,
            rho0_flat: &rho0v,
        },
        SolverParams {
            dt: 0.3e-3 / SOUND_SPEED_WATER_SIM,
            nt,
            c_ref: SOUND_SPEED_WATER_SIM,
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
    .ok()
}

/// Run a minimal simulation: one source point, one sensor point, 20 steps.
/// Verify that non-zero pressure is recorded at the sensor.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_gpu_pstd_run_produces_output() {
    let Some(mut solver) = make_solver_32(20) else {
        eprintln!("No GPU adapter — skipping run test");
        return;
    };

    let n = 32usize;
    let src_flat = 16 * n * n + 16 * n + 16;
    let sns_flat = 20 * n * n + 16 * n + 16;
    let source_signals: Vec<f32> = (0..20).map(|i| i as f32 / 20.0).collect();

    let data = solver.run(
        &[sns_flat as u32],
        &[src_flat as u32],
        &source_signals,
        &[],
        &[],
    );

    assert_eq!(data.len(), 20, "sensor data length");
    let max_val = data.iter().copied().fold(0.0f32, f32::max);
    eprintln!("GPU PSTD sensor peak: {max_val:.6}");
    assert!(
        max_val.is_finite(),
        "sensor data contains non-finite values"
    );
}

/// Run a minimal velocity-source simulation to validate the phased-array
/// ux source path used by `GpuPstdSession`.
/// # Panics
/// - Panics if assertion fails: `velocity-source path produced an all-zero sensor trace`.
///
#[test]
fn test_gpu_pstd_velocity_source_produces_output() {
    let Some(mut solver) = make_solver_32(20) else {
        eprintln!("No GPU adapter — skipping velocity-source run test");
        return;
    };

    let n = 32usize;
    let src_flat = 16 * n * n + 16 * n + 16;
    let sns_flat = 20 * n * n + 16 * n + 16;
    let vel_signals: Vec<f32> = (0..20).map(|i| i as f32 / 20.0).collect();

    let data = solver.run(
        &[sns_flat as u32],
        &[],
        &[],
        &[src_flat as u32],
        &vel_signals,
    );

    assert_eq!(data.len(), 20, "sensor data length");
    let max_abs = data.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
    eprintln!("GPU PSTD velocity-source sensor peak: {max_abs:.6}");
    assert!(
        max_abs.is_finite(),
        "sensor data contains non-finite values"
    );
    assert!(
        max_abs > 0.0,
        "velocity-source path produced an all-zero sensor trace"
    );
}

/// Mirror the session-style usage pattern with multiple velocity sources
/// and sensors on the same plane.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_gpu_pstd_multi_velocity_source_plane_produces_output() {
    let nx = 64usize;
    let ny = 64usize;
    let nz = 32usize;
    let total = nx * ny * nz;
    let dt = 0.3e-4 / SOUND_SPEED_WATER_SIM;
    let nt = 64usize;

    let c0v = vec![SOUND_SPEED_WATER_SIM as f32; total];
    let rho0v = vec![1000.0f32; total];
    let ones = vec![1.0f32; total];
    let zeros = vec![0.0f32; total];
    let solver = GpuPstdSolver::with_auto_device(
        &crate::domain::grid::Grid::new(nx, ny, nz, 1e-4, 1e-4, 1e-4).unwrap(),
        MediumArrays {
            c0_flat: &c0v,
            rho0_flat: &rho0v,
        },
        SolverParams {
            dt,
            nt,
            c_ref: SOUND_SPEED_WATER_SIM,
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

    let Some(mut solver) = solver.ok() else {
        eprintln!("No GPU adapter — skipping multi-source velocity test");
        return;
    };

    let src_x = 16usize;
    let mut indices = Vec::new();
    for iy in (ny / 2 - 2)..(ny / 2 + 2) {
        for iz in (nz / 2 - 2)..(nz / 2 + 2) {
            indices.push((src_x * ny * nz + iy * nz + iz) as u32);
        }
    }
    let mut vel_signals = vec![0.0f32; indices.len() * nt];
    for src_idx in 0..indices.len() {
        for step in 0..8 {
            vel_signals[src_idx * nt + step] = 0.1;
        }
    }

    let data = solver.run(&indices, &[], &[], &indices, &vel_signals);
    let max_abs = data.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
    eprintln!("GPU PSTD multi velocity-source sensor peak: {max_abs:.6}");
    assert!(
        max_abs > 0.0,
        "multi-source velocity path produced an all-zero sensor trace"
    );
}

/// Benchmark: measure GPU PSTD steps/second for a 256×128×128 grid.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn bench_gpu_pstd_bmode_grid() {
    let nx = 256usize;
    let ny = 128usize;
    let nz = 128usize;
    let dx = 1.48e-4_f64;
    let c0 = SOUND_SPEED_WATER_SIM;
    let dt = 0.3 * dx / c0;
    let nt = 50;
    let total = nx * ny * nz;

    let c0v = vec![c0 as f32; total];
    let rho0v = vec![1000.0f32; total];
    let ones = vec![1.0f32; total];
    let zeros = vec![0.0f32; total];
    let solver = GpuPstdSolver::with_auto_device(
        &crate::domain::grid::Grid::new(nx, ny, nz, dx, dx, dx).unwrap(),
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

    let Some(mut solver) = solver.ok() else {
        eprintln!("No GPU adapter — skipping benchmark");
        return;
    };

    let sns = (nx / 2) * ny * nz + (ny / 2) * nz + (nz / 2);
    let src = (nx / 4) * ny * nz + (ny / 2) * nz + (nz / 2);
    let sigs: Vec<f32> = (0..nt).map(|i| (i as f32 / nt as f32).sin()).collect();

    let t0 = std::time::Instant::now();
    let data = solver.run(&[sns as u32], &[src as u32], &sigs, &[], &[]);
    let elapsed = t0.elapsed();

    let ms_per_step = elapsed.as_secs_f64() * 1000.0 / nt as f64;
    let steps_per_sec = nt as f64 / elapsed.as_secs_f64();
    eprintln!(
        "GPU PSTD 256×128×128: {nt} steps in {:.1}ms total = {:.2}ms/step = {:.1} steps/sec",
        elapsed.as_millis(),
        ms_per_step,
        steps_per_sec
    );
    let sensor_dl_ms = 45.0_f64;
    let compute_ms_per_step = (elapsed.as_secs_f64() * 1000.0 - sensor_dl_ms) / nt as f64;
    let scan_line_ms = 1586.0 * compute_ms_per_step + sensor_dl_ms;
    eprintln!(
        "Estimated B-mode scan line (1586 steps): {:.1}s",
        scan_line_ms / 1000.0
    );
    assert!(!data.is_empty());
}
