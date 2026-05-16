//! GPU buffer allocation, command encoding, and readback for MVDR beamforming.

#[cfg(feature = "gpu")]
use super::params::MvdrGpuParams;
#[cfg(feature = "gpu")]
use super::MvdrGPU;
#[cfg(feature = "gpu")]
use crate::core::error::KwaversResult;
#[cfg(feature = "gpu")]
use ndarray::{Array3, Array4};
#[cfg(feature = "gpu")]
use wgpu::util::DeviceExt;

/// Submit a single MVDR GPU pass and read back the reconstructed volume.
///
/// ## Steps
/// 1. Flatten the host `rf_data` (C-contiguous row-major) into a GPU storage buffer.
/// 2. Allocate an uninitialised output buffer (every voxel is written by the shader).
/// 3. Pack `MvdrGpuParams` into a uniform buffer.
/// 4. Build a bind group (bindings 0/1/2 = RF / output / params).
/// 5. Dispatch `vol_x × vol_y × vol_z` workgroups (workgroup_size = 1×1×1).
/// 6. Copy output to a staging buffer; map synchronously; convert to `Array3<f32>`.
///
/// ## Complexity note
/// Each invocation performs O(Q · L² · N + L³) work (Q = subarrays, L ≤ 32,
/// N = num_samples).  For large N or large volumes, GPU TDR may occur on
/// devices with a 2-second watchdog.  Restrict `num_samples` and `volume_dims`
/// accordingly or disable OS TDR before long MVDR runs.
/// # Errors
/// - Propagates GPU buffer/device errors through [`KwaversResult`].
///
#[cfg(feature = "gpu")]
pub(super) fn mvdr_dispatch(
    ctx: &MvdrGPU<'_>,
    rf_data: &Array4<f32>,
    diagonal_loading: f32,
    subarray_size: [usize; 3],
) -> KwaversResult<Array3<f32>> {
    let (frames, _channels, samples, _) = rf_data.dim();
    let (vol_x, vol_y, vol_z) = ctx.config.volume_dims;
    let (nel_x, nel_y, nel_z) = ctx.config.num_elements_3d;
    let [sub_x, sub_y, sub_z] = subarray_size;

    // Ensure C-contiguous layout before casting to bytes.
    let rf_contiguous = rf_data.as_standard_layout().into_owned();
    let rf_flat: &[f32] = rf_contiguous.as_slice().expect("C-contiguous layout");

    let rf_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MVDR RF Data"),
            contents: bytemuck::cast_slice(rf_flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

    let n_voxels = vol_x * vol_y * vol_z;
    let out_bytes = (n_voxels * std::mem::size_of::<f32>()) as u64;

    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MVDR Output Volume"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = MvdrGpuParams {
        vol_x: vol_x as u32,
        vol_y: vol_y as u32,
        vol_z: vol_z as u32,
        nel_x: nel_x as u32,
        nel_y: nel_y as u32,
        nel_z: nel_z as u32,
        sub_x: sub_x as u32,
        sub_y: sub_y as u32,
        sub_z: sub_z as u32,
        num_frames: frames as u32,
        num_samples: samples as u32,
        pad0: 0,
        vox_dx: ctx.config.voxel_spacing.0 as f32,
        vox_dy: ctx.config.voxel_spacing.1 as f32,
        vox_dz: ctx.config.voxel_spacing.2 as f32,
        elem_sx: ctx.config.element_spacing_3d.0 as f32,
        elem_sy: ctx.config.element_spacing_3d.1 as f32,
        elem_sz: ctx.config.element_spacing_3d.2 as f32,
        sound_speed: ctx.config.sound_speed as f32,
        sampling_freq: ctx.config.sampling_frequency as f32,
        diagonal_loading,
        pad1: 0.0,
        pad2: 0.0,
        pad3: 0.0,
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MVDR Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("MVDR Bind Group"),
        layout: ctx.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: rf_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("MVDR Encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MVDR Compute Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(ctx.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // One workgroup per voxel (workgroup_size = 1×1×1 in the shader).
        pass.dispatch_workgroups(vol_x as u32, vol_y as u32, vol_z as u32);
    }

    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MVDR Staging"),
        size: out_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, out_bytes);
    ctx.queue.submit(Some(encoder.finish()));

    // Block until the GPU finishes.
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = ctx.device.poll(wgpu::PollType::Wait);

    let mapped = slice.get_mapped_range();
    let result: &[f32] = bytemuck::cast_slice(&mapped);
    // Shader writes row-major: output[vx * vol_y * vol_z + vy * vol_z + vz].
    // Array3::from_shape_fn with C-order matches this index.
    let volume = Array3::from_shape_fn((vol_x, vol_y, vol_z), |(x, y, z)| {
        result[x * vol_y * vol_z + y * vol_z + z]
    });
    drop(mapped);
    staging.unmap();

    Ok(volume)
}
