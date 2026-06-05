//! Test helpers for GPU PSTD solver tests.

use super::super::{AbsorptionArrays, GpuPstdSolver, MediumArrays, PmlArrays, SolverParams};
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use std::sync::Arc;

pub(super) fn make_small_test_solver() -> Option<GpuPstdSolver> {
    let nx = 8usize;
    let ny = 8usize;
    let nz = 8usize;
    let total = nx * ny * nz;
    let dx = 1.0e-3_f64;
    let c0 = SOUND_SPEED_WATER_SIM;
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

    let grid = kwavers_grid::Grid::new(nx, ny, nz, dx, dx, dx).ok()?;
    let c0_flat = vec![c0 as f32; total];
    let rho0_flat = vec![1000.0f32; total];
    let pml = vec![0.0f32; total];
    let bon_a = vec![0.375f32; total];

    GpuPstdSolver::new(
        Arc::new(device),
        Arc::new(queue),
        &grid,
        MediumArrays {
            c0_flat: &c0_flat,
            rho0_flat: &rho0_flat,
        },
        SolverParams {
            dt,
            nt,
            c_ref: c0,
            nonlinear: false,
            absorbing: false,
        },
        PmlArrays {
            x: &pml,
            y: &pml,
            z: &pml,
            sgx: &pml,
            sgy: &pml,
            sgz: &pml,
        },
        AbsorptionArrays {
            bon_a_flat: &bon_a,
            nabla1: &pml,
            nabla2: &pml,
            tau: &pml,
            eta: &pml,
        },
    )
    .ok()
}
/// Read a typed GPU buffer back to the CPU via a staging buffer.
/// # Panics
/// - Panics if `buffer readback callback`.
/// - Panics if `buffer map`.
///
pub(super) fn read_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    len: usize,
) -> Vec<T> {
    let byte_size = (len * std::mem::size_of::<T>()) as u64;
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
