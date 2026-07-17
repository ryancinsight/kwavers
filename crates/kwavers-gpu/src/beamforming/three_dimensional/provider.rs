//! WGPU provider implementation for 3-D beamforming.

use super::delay_sum::{DelaySumGPU, DynamicFocusGPU};
use super::shaders;
use hephaestus_core::{ComputeDeviceAcquisition, DeviceLimits, DevicePreference};
use hephaestus_wgpu::WgpuDevice;
use kwavers_analysis::signal_processing::beamforming::three_dimensional::{
    Beamforming3dApodizationWindow, BeamformingConfig3D, BeamformingGpuProvider,
};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_solver::backend::traits::GpuProvider;
use leto::{Array3, Array4};

/// Hephaestus-backed WGPU implementation of the 3-D beamforming provider.
#[derive(Debug)]
pub struct WgpuBeamformingProvider {
    device: WgpuDevice,
    delay_sum_pipeline: wgpu::ComputePipeline,
    dynamic_focus_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    dynamic_focus_bind_group_layout: wgpu::BindGroupLayout,
}

impl WgpuBeamformingProvider {
    /// Acquire a WGPU device through Hephaestus and compile 3-D beamforming
    /// pipelines.
    ///
    /// # Errors
    ///
    /// Returns a GPU error when WGPU acquisition fails or the selected device
    /// cannot satisfy the 512-invocation WGSL workgroup requirement.
    pub fn new() -> KwaversResult<Self> {
        let device = WgpuDevice::try_acquire_device(
            "kwavers-3d-beamforming-wgpu",
            DevicePreference::HighPerformance,
            &[],
            beamforming_required_limits(),
        )
        .map_err(|err| {
            KwaversError::System(kwavers_core::error::SystemError::ResourceUnavailable {
                resource: format!("Hephaestus WGPU 3-D beamforming device: {err}"),
            })
        })?;

        Self::from_device(device)
    }

    fn from_device(device: WgpuDevice) -> KwaversResult<Self> {
        let raw_device = device.device().as_ref();

        let delay_sum_shader = raw_device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("3D Delay-and-Sum Shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::BEAMFORMING_3D_SHADER.into()),
        });

        let dynamic_focus_shader = raw_device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("3D Dynamic-Focus Shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::DYNAMIC_FOCUS_3D_SHADER.into()),
        });

        let bind_group_layout =
            raw_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("3D Beamforming Bind Group Layout"),
                entries: &[
                    storage_ro(0),
                    storage_rw(1),
                    uniform(2),
                    storage_ro(3),
                    storage_ro(4),
                ],
            });

        let dynamic_focus_bind_group_layout =
            raw_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Dynamic-Focus Bind Group Layout"),
                entries: &[
                    storage_ro(0),
                    storage_rw(1),
                    uniform(2),
                    storage_ro(3),
                    storage_ro(4),
                    storage_ro(5),
                    storage_ro(6),
                ],
            });

        let pipeline_layout = raw_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("3D Beamforming Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let dynamic_focus_pipeline_layout =
            raw_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Dynamic-Focus Pipeline Layout"),
                bind_group_layouts: &[Some(&dynamic_focus_bind_group_layout)],
                immediate_size: 0,
            });

        let delay_sum_pipeline =
            raw_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("3D Delay-and-Sum Pipeline"),
                layout: Some(&pipeline_layout),
                module: &delay_sum_shader,
                entry_point: Some("delay_and_sum_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let dynamic_focus_pipeline =
            raw_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("3D Dynamic-Focus Pipeline"),
                layout: Some(&dynamic_focus_pipeline_layout),
                module: &dynamic_focus_shader,
                entry_point: Some("dynamic_focus_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Ok(Self {
            device,
            delay_sum_pipeline,
            dynamic_focus_pipeline,
            bind_group_layout,
            dynamic_focus_bind_group_layout,
        })
    }
}

impl BeamformingGpuProvider for WgpuBeamformingProvider {
    fn provider_kind(&self) -> GpuProvider {
        GpuProvider::Wgpu
    }

    fn process_delay_and_sum(
        &self,
        config: &BeamformingConfig3D,
        rf_data: &Array4<f32>,
        dynamic_focusing: bool,
        apodization_window: &Beamforming3dApodizationWindow,
        apodization_weights: &Array3<f32>,
        sub_volume_size: Option<(usize, usize, usize)>,
    ) -> KwaversResult<Array3<f32>> {
        if dynamic_focusing {
            let dynamic_focus = DynamicFocusGPU {
                config,
                device: self.device.device().as_ref(),
                queue: self.device.queue().as_ref(),
                pipeline: &self.dynamic_focus_pipeline,
                bind_group_layout: &self.dynamic_focus_bind_group_layout,
            };
            return dynamic_focus.process(rf_data, apodization_window, apodization_weights);
        }

        let delay_sum = DelaySumGPU::new(
            config,
            self.device.device().as_ref(),
            self.device.queue().as_ref(),
            &self.delay_sum_pipeline,
            &self.bind_group_layout,
        );

        match sub_volume_size {
            Some(sub_size) => delay_sum.process_subvolume(
                rf_data,
                dynamic_focusing,
                apodization_window,
                apodization_weights,
                sub_size,
            ),
            None => delay_sum.process(
                rf_data,
                dynamic_focusing,
                apodization_window,
                apodization_weights,
            ),
        }
    }
}

fn beamforming_required_limits() -> DeviceLimits {
    let base = WgpuDevice::default_device_limits();
    DeviceLimits {
        max_buffer_size: base.max_buffer_size,
        max_compute_workgroup_size_x: base.max_compute_workgroup_size_x.max(8),
        max_compute_workgroup_size_y: base.max_compute_workgroup_size_y.max(8),
        max_compute_workgroup_size_z: base.max_compute_workgroup_size_z.max(8),
        max_compute_invocations_per_workgroup: base.max_compute_invocations_per_workgroup.max(512),
        max_compute_workgroup_storage_size: base.max_compute_workgroup_storage_size,
        max_storage_buffers_per_shader_stage: base.max_storage_buffers_per_shader_stage,
        max_immediate_size: base.max_immediate_size,
    }
}

fn storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    storage(binding, true)
}

fn storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    storage(binding, false)
}

fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::WgpuBeamformingProvider;
    use kwavers_analysis::signal_processing::beamforming::three_dimensional::{
        delay_and_sum_cpu_reference, Beamforming3dApodizationWindow, BeamformingAlgorithm3D,
        BeamformingConfig3D, BeamformingProcessor3D,
    };
    use leto::Array4;

    #[test]
    fn wgpu_das_matches_cpu_reference_when_available() {
        let config = BeamformingConfig3D {
            num_elements_3d: (2, 1, 2),
            element_spacing_3d: (1.0e-3, 1.0e-3, 1.0e-3),
            volume_dims: (4, 1, 4),
            voxel_spacing: (1.0e-3, 1.0e-3, 1.0e-3),
            sound_speed: 1500.0,
            sampling_frequency: 10.0e6,
            ..BeamformingConfig3D::default()
        };

        let mut rf = Array4::<f32>::zeros((1, 4, 64, 1));
        for ch in 0..4 {
            for s in 0..64 {
                rf[[0, ch, s, 0]] = (((ch * 13 + s * 7) % 17) as f32) - 8.0;
            }
        }

        let cpu =
            delay_and_sum_cpu_reference(&rf, &config, &Beamforming3dApodizationWindow::Rectangular)
                .expect("CPU DAS reference must accept the deterministic fixture");

        let Ok(provider) = WgpuBeamformingProvider::new() else {
            return;
        };
        let mut processor = BeamformingProcessor3D::with_provider(config, provider)
            .expect("WGPU provider construction already succeeded");

        let gpu = processor
            .process_volume(
                &rf,
                &BeamformingAlgorithm3D::DelayAndSum {
                    dynamic_focusing: false,
                    apodization: Beamforming3dApodizationWindow::Rectangular,
                    sub_volume_size: None,
                },
            )
            .expect("WGPU DAS reconstruction must dispatch when provider is available");

        assert_eq!(cpu.shape(), gpu.shape(), "CPU/GPU volume dimensions differ");
        for (c, g) in cpu.iter().zip(gpu.iter()) {
            let tol = 1.0e-3f32.mul_add(c.abs(), 1.0e-2);
            assert!(
                (c - g).abs() <= tol,
                "WGPU DAS diverges from CPU reference: cpu={c}, gpu={g}, tol={tol}"
            );
        }
    }
}
