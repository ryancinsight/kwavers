//! Dynamic-focus GPU dispatch for [`DynamicFocusGPU`].
//!
//! # Algorithm
//!
//! 1. CPU pre-computes focus delays: for each focus zone *z* and each element
//!    *e*, the delay (in samples) is
//!    ```text
//!    delay[z, e] = |focal_pos(z) − element_pos(e)| / c · f_s
//!    ```
//!    where the focal point for zone *z* is `(0, 0, min_depth + z·Δz)` and
//!    `Δz = (max_depth − min_depth) / (num_zones − 1)`.
//! 2. CPU uploads the delay table and a zeroed aperture-mask buffer (variable
//!    aperture is disabled at this tier; the mask buffer must be bound even
//!    though the shader ignores it when `enable_variable_aperture == 0`).
//! 3. GPU runs `dynamic_focus_main` (workgroup 8×8×8), reads the delay table
//!    from the bound storage buffer, and coherently sums delayed RF samples.
//!
//! # References
//! - Synnevåg, Austeng & Holm (2007): Adaptive beamforming applied to medical ultrasound imaging.
//! - Jensen (1996): Field: A Program for Simulating Ultrasound Systems.

#[cfg(feature = "gpu")]
use super::super::params::DynamicFocusParams;
#[cfg(feature = "gpu")]
use kwavers_analysis::signal_processing::beamforming::three_dimensional::{
    Beamforming3dApodizationWindow, BeamformingConfig3D,
};
#[cfg(feature = "gpu")]
use kwavers_core::error::KwaversResult;
#[cfg(feature = "gpu")]
use leto::{
    Array3,
    Array4,
};
#[cfg(feature = "gpu")]
use wgpu::util::DeviceExt;

/// Default F-number (depth / aperture) for dynamic focusing.
#[cfg(feature = "gpu")]
const DEFAULT_F_NUMBER: f32 = 1.5;
/// Default number of depth-stratified focus zones.
#[cfg(feature = "gpu")]
const DEFAULT_NUM_FOCUS_ZONES: u32 = 32;

/// GPU dynamic-focus delay-and-sum processor.
///
/// Borrows the device, queue, pipeline, and bind group layout that
/// `BeamformingProcessor3D` owns for the dynamic-focus path.
#[cfg(feature = "gpu")]
pub struct DynamicFocusGPU<'a> {
    pub(crate) config: &'a BeamformingConfig3D,
    pub(crate) device: &'a wgpu::Device,
    pub(crate) queue: &'a wgpu::Queue,
    pub(crate) pipeline: &'a wgpu::ComputePipeline,
    pub(crate) bind_group_layout: &'a wgpu::BindGroupLayout,
}

#[cfg(feature = "gpu")]
impl<'a> DynamicFocusGPU<'a> {
    /// Execute dynamic-focus delay-and-sum beamforming on GPU.
    ///
    /// `min_depth` / `max_depth` define the axial range (metres) across which
    /// `num_focus_zones` focal positions are uniformly distributed.  If these
    /// parameters are not supplied via the calling context, sensible defaults
    /// are derived from the volume geometry.
    ///
    /// # Errors
    /// - Propagates `KwaversError::System` on GPU device errors.
    pub fn process(
        &self,
        rf_data: &Array4<f32>,
        apodization_window: &Beamforming3dApodizationWindow,
        apodization_weights: &Array3<f32>,
    ) -> KwaversResult<Array3<f32>> {
        let rf_dims = rf_data.dim();
        let frames = rf_dims.0;
        let samples = rf_dims.2;

        let (vol_x, vol_y, vol_z) = (
            self.config.volume_dims.0,
            self.config.volume_dims.1,
            self.config.volume_dims.2,
        );

        // Derive focus range from volume geometry.
        let min_depth = 0.0f32;
        let max_depth = vol_z as f32 * self.config.voxel_spacing.2 as f32;
        let num_focus_zones = DEFAULT_NUM_FOCUS_ZONES;
        let total_elements = self.config.num_elements_3d.0
            * self.config.num_elements_3d.1
            * self.config.num_elements_3d.2;

        // --- CPU pre-computation of delay table [num_focus_zones × total_elements] ---
        let element_positions = self.create_element_positions();
        // element_positions is flat [x0,y0,z0, x1,y1,z1, …] → repack as [(x,y,z)]
        let elem_pos: Vec<[f32; 3]> = (0..total_elements)
            .map(|i| {
                [
                    element_positions[3 * i],
                    element_positions[3 * i + 1],
                    element_positions[3 * i + 2],
                ]
            })
            .collect();

        let mut focus_delays = vec![0.0f32; num_focus_zones as usize * total_elements];
        let c = self.config.sound_speed as f32;
        let fs = self.config.sampling_frequency as f32;
        for zone in 0..num_focus_zones as usize {
            let depth = if num_focus_zones > 1 {
                min_depth + zone as f32 * (max_depth - min_depth) / (num_focus_zones - 1) as f32
            } else {
                (min_depth + max_depth) * 0.5
            };
            let focus = [0.0f32, 0.0, depth];
            for (elem_idx, ep) in elem_pos.iter().enumerate() {
                let dx = focus[0] - ep[0];
                let dy = focus[1] - ep[1];
                let dz = focus[2] - ep[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                focus_delays[zone * total_elements + elem_idx] = dist / c * fs;
            }
        }

        // Aperture-mask buffer (zeroed — variable aperture is disabled).
        let aperture_mask_elements = num_focus_zones as usize * total_elements.div_ceil(32);
        let aperture_masks = vec![0u32; aperture_mask_elements];

        // --- GPU buffer uploads ---
        let rf_flat: Vec<f32> = rf_data.as_slice().unwrap_or(&[]).to_vec();
        let rf_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("DF RF Data"),
                contents: bytemuck::cast_slice(&rf_flat),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let output_flat = vec![0.0f32; vol_x * vol_y * vol_z];
        let output_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("DF Output Volume"),
                contents: bytemuck::cast_slice(&output_flat),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let apodization_flat: Vec<f32> = apodization_weights.as_slice().unwrap_or(&[]).to_vec();
        let apodization_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("DF Apodization"),
                    contents: bytemuck::cast_slice(&apodization_flat),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let element_positions_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("DF Element Positions"),
                    contents: bytemuck::cast_slice(&element_positions),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let focus_delays_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("DF Focus Delays"),
                    contents: bytemuck::cast_slice(&focus_delays),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let aperture_masks_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("DF Aperture Masks"),
                    contents: bytemuck::cast_slice(&aperture_masks),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let apodization_window_u32 = match apodization_window {
            Beamforming3dApodizationWindow::Rectangular => 0,
            Beamforming3dApodizationWindow::Hamming => 1,
            Beamforming3dApodizationWindow::Hann => 2,
            Beamforming3dApodizationWindow::Blackman => 3,
            _ => 0,
        };
        let _ = apodization_window_u32; // used only for static DAS Params; DF shader uses weights directly

        let params = DynamicFocusParams {
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
            sound_speed: c,
            sampling_freq: fs,
            center_freq: self.config.center_frequency as f32,
            f_number: DEFAULT_F_NUMBER,
            min_depth,
            max_depth,
            num_focus_zones,
            _padding5: 0,
            num_frames: frames as u32,
            num_samples: samples as u32,
            enable_variable_aperture: 0,
            _padding6: 0,
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("DF Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dynamic Focus Bind Group"),
            layout: self.bind_group_layout,
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: apodization_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: element_positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: focus_delays_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: aperture_masks_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Dynamic Focus Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Dynamic Focus Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // Must match @workgroup_size in dynamic_focus_3d.wgsl (8³ = 512).
            let ws = 8_usize;
            pass.dispatch_workgroups(
                vol_x.div_ceil(ws) as u32,
                vol_y.div_ceil(ws) as u32,
                vol_z.div_ceil(ws) as u32,
            );
        }

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DF Staging"),
            size: (vol_x * vol_y * vol_z * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, staging.size());
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::Wait);

        let mapped = slice.get_mapped_range();
        let result_f32: &[f32] = bytemuck::cast_slice(&mapped);
        let result = Array3::from_shape_fn((vol_x, vol_y, vol_z), |(x, y, z)| {
            result_f32[x + y * vol_x + z * vol_x * vol_y]
        });
        staging.unmap();

        Ok(result)
    }

    /// Build element positions flat buffer — reuses the same geometry as static DAS.
    fn create_element_positions(&self) -> Vec<f32> {
        let mut positions = Vec::new();
        for ex in 0..self.config.num_elements_3d.0 {
            for ey in 0..self.config.num_elements_3d.1 {
                for ez in 0..self.config.num_elements_3d.2 {
                    let x = ((self.config.num_elements_3d.0 - 1) as f32).mul_add(-0.5, ex as f32)
                        * self.config.element_spacing_3d.0 as f32;
                    let y = ((self.config.num_elements_3d.1 - 1) as f32).mul_add(-0.5, ey as f32)
                        * self.config.element_spacing_3d.1 as f32;
                    let z = ((self.config.num_elements_3d.2 - 1) as f32).mul_add(-0.5, ez as f32)
                        * self.config.element_spacing_3d.2 as f32;
                    positions.extend_from_slice(&[x, y, z]);
                }
            }
        }
        positions
    }
}
