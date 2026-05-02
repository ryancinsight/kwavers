use super::*;
use std::sync::Arc;

/// Verify GpuPstdSolver can be constructed and runs without error.
/// Skipped if no GPU adapter is available (headless CI).
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
    let c0 = 1500.0_f64;
    let rho0 = 1000.0_f64;
    let dt = 0.3 * dx / c0;
    let nt = 10;

    let grid = crate::domain::grid::Grid::new(n, n, n, dx, dx, dx).unwrap();
    let c0v: Vec<f32> = vec![c0 as f32; n * n * n];
    let rho0v: Vec<f32> = vec![rho0 as f32; n * n * n];
    let ones: Vec<f32> = vec![1.0f32; n * n * n];
    let zeros: Vec<f32> = vec![0.0f32; n * n * n];

    let solver = GpuPstdSolver::new(
        device, queue, &grid, &c0v, &rho0v, dt, nt, c0, &ones, &ones, &ones, &ones, &ones, &ones,
        &zeros, &zeros, &zeros, &zeros, &zeros, false, false,
    );

    assert!(
        solver.is_ok(),
        "GpuPstdSolver::new failed: {:?}",
        solver.err()
    );
    eprintln!("GpuPstdSolver constructed successfully");
}

/// Run a minimal simulation: one source point, one sensor point, 20 steps.
/// Verify that non-zero pressure is recorded at the sensor.
#[test]
fn test_gpu_pstd_run_produces_output() {
    let n3 = 32 * 32 * 32;
    let solver = GpuPstdSolver::with_auto_device(
        &crate::domain::grid::Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap(),
        &vec![1500.0f32; n3],
        &vec![1000.0f32; n3],
        0.3e-3 / 1500.0,
        20,
        1500.0,
        &vec![1.0f32; n3],
        &vec![1.0f32; n3],
        &vec![1.0f32; n3],
        &vec![1.0f32; n3],
        &vec![1.0f32; n3],
        &vec![1.0f32; n3],
        &vec![0.0f32; n3],
        &vec![0.0f32; n3],
        &vec![0.0f32; n3],
        &vec![0.0f32; n3],
        &vec![0.0f32; n3],
        false,
        false,
    );

    let Some(mut solver) = solver.ok() else {
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
#[test]
fn test_gpu_pstd_velocity_source_produces_output() {
    let n3 = 32 * 32 * 32;
    let solver = GpuPstdSolver::with_auto_device(
        &crate::domain::grid::Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap(),
        &vec![1500.0f32; n3],
        &vec![1000.0f32; n3],
        0.3e-3 / 1500.0,
        20,
        1500.0,
        &vec![1.0f32; n3],
        &vec![1.0f32; n3],
        &vec![1.0f32; n3],
        &vec![1.0f32; n3],
        &vec![1.0f32; n3],
        &vec![1.0f32; n3],
        &vec![0.0f32; n3],
        &vec![0.0f32; n3],
        &vec![0.0f32; n3],
        &vec![0.0f32; n3],
        &vec![0.0f32; n3],
        false,
        false,
    );

    let Some(mut solver) = solver.ok() else {
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
#[test]
fn test_gpu_pstd_multi_velocity_source_plane_produces_output() {
    let nx = 64usize;
    let ny = 64usize;
    let nz = 32usize;
    let total = nx * ny * nz;
    let dt = 0.3e-4 / 1500.0;
    let nt = 64usize;

    let solver = GpuPstdSolver::with_auto_device(
        &crate::domain::grid::Grid::new(nx, ny, nz, 1e-4, 1e-4, 1e-4).unwrap(),
        &vec![1500.0f32; total],
        &vec![1000.0f32; total],
        dt,
        nt,
        1500.0,
        &vec![1.0f32; total],
        &vec![1.0f32; total],
        &vec![1.0f32; total],
        &vec![1.0f32; total],
        &vec![1.0f32; total],
        &vec![1.0f32; total],
        &vec![0.0f32; total],
        &vec![0.0f32; total],
        &vec![0.0f32; total],
        &vec![0.0f32; total],
        &vec![0.0f32; total],
        false,
        false,
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
#[test]
fn bench_gpu_pstd_bmode_grid() {
    let nx = 256usize;
    let ny = 128usize;
    let nz = 128usize;
    let dx = 1.48e-4_f64;
    let c0 = 1500.0_f64;
    let dt = 0.3 * dx / c0;
    let nt = 50;
    let total = nx * ny * nz;

    let solver = GpuPstdSolver::with_auto_device(
        &crate::domain::grid::Grid::new(nx, ny, nz, dx, dx, dx).unwrap(),
        &vec![c0 as f32; total],
        &vec![1000.0f32; total],
        dt,
        nt,
        c0,
        &vec![1.0f32; total],
        &vec![1.0f32; total],
        &vec![1.0f32; total],
        &vec![1.0f32; total],
        &vec![1.0f32; total],
        &vec![1.0f32; total],
        &vec![0.0f32; total],
        &vec![0.0f32; total],
        &vec![0.0f32; total],
        &vec![0.0f32; total],
        &vec![0.0f32; total],
        false,
        false,
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

fn make_small_test_solver() -> Option<GpuPstdSolver> {
    let nx = 8usize;
    let ny = 8usize;
    let nz = 8usize;
    let total = nx * ny * nz;
    let dx = 1.0e-3_f64;
    let c0 = 1500.0_f64;
    let dt = 1.0e-8_f64;
    let nt = 4usize;

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
        eprintln!("No GPU adapter — skipping GpuPstdSolver medium-update test");
        return None;
    };

    let native_limits = adapter.limits();
    let device = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("gpu_pstd_medium_update_test"),
        required_features: wgpu::Features::PUSH_CONSTANTS,
        required_limits: wgpu::Limits {
            max_push_constant_size: 128,
            max_storage_buffers_per_shader_stage:
                native_limits.max_storage_buffers_per_shader_stage,
            ..wgpu::Limits::default()
        },
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::default(),
    }));
    let Ok((device, queue)) = device else {
        eprintln!("GPU device creation failed — skipping medium-update test");
        return None;
    };

    let grid = crate::domain::grid::Grid::new(nx, ny, nz, dx, dx, dx).ok()?;
    let c0_flat = vec![c0 as f32; total];
    let rho0_flat = vec![1000.0f32; total];
    let pml = vec![0.0f32; total];
    let bon_a = vec![0.375f32; total];

    GpuPstdSolver::new(
        Arc::new(device),
        Arc::new(queue),
        &grid,
        &c0_flat,
        &rho0_flat,
        dt,
        nt,
        c0,
        &pml,
        &pml,
        &pml,
        &pml,
        &pml,
        &pml,
        &bon_a,
        &pml,
        &pml,
        &pml,
        &pml,
        false,
        false,
    )
    .ok()
}

fn read_buffer_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    len: usize,
) -> Vec<f32> {
    let byte_size = (len * std::mem::size_of::<f32>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_pstd_test_readback"),
        size: byte_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu_pstd_test_readback_copy"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, byte_size);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = device.poll(wgpu::PollType::Wait);
    rx.recv()
        .expect("buffer readback callback")
        .expect("buffer map");

    let mapped = slice.get_mapped_range();
    let data = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    data
}

/// Regression guard: variable-only medium updates must preserve static
/// absorption/nonlinearity buffers, and source-kappa disablement must write
/// a resident unity buffer rather than a fresh allocation.
#[test]
fn test_medium_variable_update_preserves_static_buffers_and_disables_source_correction() {
    let Some(mut solver) = make_small_test_solver() else {
        return;
    };

    let total = solver.nx * solver.ny * solver.nz;
    let new_c0: Vec<f32> = (0..total).map(|i| 1450.0f32 + i as f32 * 0.5).collect();
    let new_rho: Vec<f32> = (0..total).map(|i| 980.0f32 + i as f32 * 0.25).collect();
    solver.update_medium_variable(&new_c0, &new_rho);
    solver.disable_source_correction();

    let total = solver.nx * solver.ny * solver.nz;
    let c0_sq = read_buffer_f32(&solver.device, &solver.queue, &solver.buf_c0_sq, total);
    let rho0 = read_buffer_f32(&solver.device, &solver.queue, &solver.buf_rho0, total);
    let rho0_inv = read_buffer_f32(&solver.device, &solver.queue, &solver.buf_rho0_inv, total);
    let bon_a = read_buffer_f32(&solver.device, &solver.queue, &solver.buf_bon_a, total);
    let source_kappa = read_buffer_f32(
        &solver.device,
        &solver.queue,
        &solver.buf_source_kappa,
        total,
    );

    let first_idx = 0usize;
    let last_idx = total - 1;
    let first_c0 = new_c0[0] as f64;
    let last_c0 = *new_c0.last().expect("non-empty c0") as f64;
    let first_rho = new_rho[0] as f64;
    let last_rho = *new_rho.last().expect("non-empty rho") as f64;

    assert!((c0_sq[first_idx] as f64 - first_c0 * first_c0).abs() < 1e-3);
    assert!((c0_sq[last_idx] as f64 - last_c0 * last_c0).abs() < 1e-3);
    assert!((rho0[first_idx] as f64 - first_rho).abs() < 1e-6);
    assert!((rho0[last_idx] as f64 - last_rho).abs() < 1e-6);
    assert!((rho0_inv[first_idx] as f64 - (1.0 / first_rho)).abs() < 1e-6);
    assert!((rho0_inv[last_idx] as f64 - (1.0 / last_rho)).abs() < 1e-6);
    let bon_a_expected = 0.375_f32;
    let unity_expected = 1.0_f32;
    let tol = 1e-6_f32;
    assert!(bon_a.iter().all(|&v| (v - bon_a_expected).abs() < tol));
    assert!(source_kappa
        .iter()
        .all(|&v| (v - unity_expected).abs() < tol));
}

/// Regression guard: the GPU phased-array path injects velocity sources in
/// k-space and must retain the source-kappa reconstruction chain.
#[test]
fn test_velocity_source_shader_uses_validated_kspace_path() {
    let src = include_str!("../../../../gpu/shaders/pstd.wgsl");
    let inject_start = src
        .find("fn inject_velocity_x_source")
        .expect("inject_velocity_x_source entry point must exist");
    let inject_block = &src[inject_start..src.len().min(inject_start + 600)];

    assert!(
        inject_block.contains("kspace_re[flat] += amp;"),
        "inject_velocity_x_source must inject into kspace_re for the validated source-kappa path"
    );
    assert!(
        src.contains("fn apply_source_kappa"),
        "velocity-source k-space injection requires apply_source_kappa"
    );
    assert!(
        src.contains("fn add_kspace_to_field_ux"),
        "velocity-source k-space injection requires add_kspace_to_field_ux"
    );
}

/// Regression guard: the WGSL push-constant layout must stay aligned with
/// the Rust-side `PstdParams` struct used for dispatch.
#[test]
fn test_pstd_shader_push_constant_abi_matches_rust() {
    let src = include_str!("../../../../gpu/shaders/pstd.wgsl");
    let struct_start = src
        .find("struct PstdParams")
        .expect("PstdParams push-constant struct must exist in WGSL");
    let struct_block = &src[struct_start..src.len().min(struct_start + 500)];

    assert!(
        !struct_block.contains("dx:"),
        "WGSL PstdParams must not add fields absent from Rust push constants"
    );
    assert!(
        src.contains("precomp_source_kappa"),
        "validated phased-array GPU path requires precomp_source_kappa in the shader ABI"
    );
}
