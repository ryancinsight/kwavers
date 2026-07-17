//! Test helpers for GPU PSTD solver tests.

use super::super::{
    AbsorptionArrays, GpuPstdSolver, MediumArrays, PmlArrays, SolverParams, WgpuPstdStateProvider,
};
use crate::pstd_gpu::state::LOSSLESS_PIPELINE_BUFFERS_PER_SHADER_STAGE;
use crate::{backend::init::GpuProviderContext, gpu::GpuDeviceProvider};
use hephaestus_core::DeviceFeature;
use hephaestus_wgpu::WgpuDevice;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;

pub(super) fn pstd_test_provider(label: &str) -> Option<GpuProviderContext<WgpuDevice>> {
    GpuProviderContext::<WgpuDevice>::with_features_and_limits(
        WgpuDevice::acquisition_preference(),
        &[DeviceFeature::ImmediateData],
        pstd_test_required_limits(),
    )
    .map_err(|error| {
        eprintln!("{label}: GPU device creation failed - {error}");
        error
    })
    .ok()
}

pub(super) fn make_small_test_solver() -> Option<GpuPstdSolver> {
    let nx = 8usize;
    let ny = 8usize;
    let nz = 8usize;
    let total = nx * ny * nz;
    let dx = 1.0e-3_f64;
    let c0 = SOUND_SPEED_WATER_SIM;
    let dt = 1.0e-8_f64;
    let nt = 4usize;

    let provider = pstd_test_provider("gpu_pstd_medium_update_test")?;

    let grid = kwavers_grid::Grid::new(nx, ny, nz, dx, dx, dx).ok()?;
    let c0_flat = vec![c0 as f32; total];
    let rho0_flat = vec![1000.0f32; total];
    let pml = vec![0.0f32; total];
    let bon_a = vec![0.375f32; total];

    GpuPstdSolver::<WgpuPstdStateProvider>::new(
        provider,
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

fn pstd_test_required_limits() -> hephaestus_core::DeviceLimits {
    hephaestus_core::DeviceLimits {
        max_storage_buffers_per_shader_stage: Some(LOSSLESS_PIPELINE_BUFFERS_PER_SHADER_STAGE),
        max_immediate_size: 128,
        ..WgpuDevice::required_limits()
    }
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
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .expect("buffer readback callback")
        .expect("buffer map");

    let mapped = slice
        .get_mapped_range()
        .expect("map_async success must expose the staging-buffer range");
    let data = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    data
}
