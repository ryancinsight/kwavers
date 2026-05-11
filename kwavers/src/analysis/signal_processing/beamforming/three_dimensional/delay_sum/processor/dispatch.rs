//! GPU dispatch methods for [`DelaySumGPU`].
//!
//! `process` submits a single GPU compute pass.
//! `process_subvolume` delegates to `process` with an explicit sub-volume hint
//! (reserved for future tiled dispatch).
//!
//! Both methods are only compiled when the `gpu` feature is enabled.

#[cfg(feature = "gpu")]
use super::DelaySumGPU;
#[cfg(feature = "gpu")]
use crate::analysis::signal_processing::beamforming::three_dimensional::config::ApodizationWindow;
#[cfg(feature = "gpu")]
use crate::core::error::KwaversResult;
#[cfg(feature = "gpu")]
use ndarray::{Array3, Array4};
#[cfg(feature = "gpu")]
use super::super::params::Params;
#[cfg(feature = "gpu")]
use wgpu::util::DeviceExt;

#[cfg(feature = "gpu")]
impl<'a> DelaySumGPU<'a> {
    /// Execute static delay-and-sum beamforming on GPU.
    ///
    /// Uploads RF data, apodization weights, and element positions to GPU
    /// storage buffers, dispatches the `delay_and_sum_main` compute shader, and
    /// reads back the reconstructed volume.
    ///
    /// `dynamic_focusing` is accepted for API symmetry but must be `false`
    /// here: the caller (`BeamformingProcessor3D::process_delay_and_sum`) routes
    /// `dynamic_focusing = true` to [`DynamicFocusGPU`] before reaching this
    /// method.
    /// # Errors
    /// - Propagates GPU device errors via `KwaversError::System`.
    ///
    pub fn process(
        &self,
        rf_data: &Array4<f32>,
        dynamic_focusing: bool,
        apodization_window: &ApodizationWindow,
        apodization_weights: &Array3<f32>,
    ) -> KwaversResult<Array3<f32>> {
        debug_assert!(
            !dynamic_focusing,
            "DelaySumGPU::process must not be called with dynamic_focusing=true; \
             route through DynamicFocusGPU instead"
        );

        let rf_dims = rf_data.dim();
        let frames = rf_dims.0;
        let _channels = rf_dims.1;
        let samples = rf_dims.2;
        let (vol_x, vol_y, vol_z) = (
            self.config.volume_dims.0,
            self.config.volume_dims.1,
            self.config.volume_dims.2,
        );

        let rf_data_flat: Vec<f32> = rf_data.as_slice().unwrap_or(&[]).to_vec();
        let rf_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RF Data Buffer"),
                contents: bytemuck::cast_slice(&rf_data_flat),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let output_volume = Array3::<f32>::zeros((vol_x, vol_y, vol_z));
        let output_flat: Vec<f32> = output_volume.as_slice().unwrap_or(&[]).to_vec();
        let output_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Output Volume Buffer"),
                    contents: bytemuck::cast_slice(&output_flat),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                });

        let apodization_flat: Vec<f32> = apodization_weights.as_slice().unwrap_or(&[]).to_vec();
        let apodization_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Apodization Weights Buffer"),
                    contents: bytemuck::cast_slice(&apodization_flat),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let element_positions = self.create_element_positions();
        let element_positions_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Element Positions Buffer"),
                    contents: bytemuck::cast_slice(&element_positions),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let apodization_window_u32 = match apodization_window {
            ApodizationWindow::Rectangular => 0,
            ApodizationWindow::Hamming => 1,
            ApodizationWindow::Hann => 2,
            ApodizationWindow::Blackman => 3,
            ApodizationWindow::Gaussian { .. } => 0,
            ApodizationWindow::Custom(_) => 0,
        };

        let params = Params {
            volume_dims: [vol_x as u32, vol_y as u32, vol_z as u32],
            _padding1: 0,
            voxel_spacing: [
                self.config.voxel_spacing.0 as f32,
                self.config.voxel_spacing.1 as f32,
                self.config.voxel_spacing.2 as f32,
            ],
            _padding2: 0,
            num_elements: [
                self.config.num_elements_3d.0 as u32,
                self.config.num_elements_3d.1 as u32,
                self.config.num_elements_3d.2 as u32,
            ],
            _padding3: 0,
            element_spacing: [
                self.config.element_spacing_3d.0 as f32,
                self.config.element_spacing_3d.1 as f32,
                self.config.element_spacing_3d.2 as f32,
            ],
            _padding4: 0,
            sound_speed: self.config.sound_speed as f32,
            sampling_freq: self.config.sampling_frequency as f32,
            center_freq: self.config.center_frequency as f32,
            _padding5: 0.0,
            num_frames: frames as u32,
            num_samples: samples as u32,
            dynamic_focusing: if dynamic_focusing { 1 } else { 0 },
            apodization_window: apodization_window_u32,
        };

        let params_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Parameters Buffer"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("3D Beamforming Bind Group"),
            layout: self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &rf_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &output_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &apodization_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &element_positions_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        let mut encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("3D Beamforming Encoder"),
                });

        {
            let mut compute_pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("3D Beamforming Pass"),
                    timestamp_writes: None,
                });

            compute_pass.set_pipeline(self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_size = 8_usize;
            let dispatch_x = vol_x.div_ceil(workgroup_size);
            let dispatch_y = vol_y.div_ceil(workgroup_size);
            let dispatch_z = vol_z.div_ceil(workgroup_size);

            compute_pass.dispatch_workgroups(
                dispatch_x as u32,
                dispatch_y as u32,
                dispatch_z as u32,
            );
        }

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (vol_x * vol_y * vol_z * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            staging_buffer.size(),
        );

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});

        let _ = self.device.poll(wgpu::PollType::Wait);

        let data = buffer_slice.get_mapped_range();
        let result_f32: &[f32] = bytemuck::cast_slice(&data);

        let result_volume = Array3::from_shape_fn((vol_x, vol_y, vol_z), |(x, y, z)| {
            let idx = x + y * vol_x + z * vol_x * vol_y;
            result_f32[idx]
        });

        staging_buffer.unmap();

        Ok(result_volume)
    }

    /// Process sub-volume for memory efficiency.
    ///
    /// Delegates to [`process`](Self::process). Tiled sub-volume dispatch
    /// (requiring delay tables and aperture mask buffers) is reserved for a
    /// future increment.
    /// # Errors
    /// - Propagates errors from [`process`](Self::process).
    ///
    pub fn process_subvolume(
        &self,
        rf_data: &Array4<f32>,
        dynamic_focusing: bool,
        apodization_window: &ApodizationWindow,
        apodization_weights: &Array3<f32>,
        _sub_volume_size: (usize, usize, usize),
    ) -> KwaversResult<Array3<f32>> {
        self.process(
            rf_data,
            dynamic_focusing,
            apodization_window,
            apodization_weights,
        )
    }
}
