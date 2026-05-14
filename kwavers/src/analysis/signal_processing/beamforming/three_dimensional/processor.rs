//! GPU-Accelerated 3D Beamforming Processor
//!
//! Core processor structure that manages GPU device initialization, compute pipelines,
//! and real-time streaming buffers for volumetric ultrasound beamforming.
//!
//! # Architecture
//! - WGPU device and queue management
//! - Compute pipeline setup for delay-and-sum and dynamic focusing
//! - Bind group layout configuration
//! - Streaming buffer management for real-time 4D imaging
//!
//! # References
//! - Jensen (1996) - Field: A Program for Simulating Ultrasound Systems
//! - Synnevåg et al. (2005) - Adaptive beamforming applied to medical ultrasound imaging

use super::config::{BeamformingConfig3D, BeamformingMetrics};
use crate::core::error::{KwaversError, KwaversResult};
#[cfg(feature = "gpu")]
use crate::domain::sensor::beamforming::shaders;

#[cfg(feature = "gpu")]
use wgpu;

/// Real-time 3D beamforming processor with optional GPU acceleration
#[derive(Debug)]
pub struct BeamformingProcessor3D {
    /// Configuration — used by both GPU and CPU execution paths.
    pub(crate) config: BeamformingConfig3D,
    #[cfg(feature = "gpu")]
    /// WGPU device
    pub(crate) device: wgpu::Device,
    #[cfg(feature = "gpu")]
    /// WGPU queue
    pub(crate) queue: wgpu::Queue,
    #[cfg(feature = "gpu")]
    /// Compute pipeline for static delay-and-sum (`beamforming_3d.wgsl`).
    pub(crate) delay_sum_pipeline: wgpu::ComputePipeline,
    #[cfg(feature = "gpu")]
    /// Compute pipeline for dynamic-focus delay-and-sum (`dynamic_focus_3d.wgsl`).
    pub(crate) dynamic_focus_pipeline: wgpu::ComputePipeline,
    #[cfg(feature = "gpu")]
    /// Bind group layout for static DAS (5 bindings).
    pub(crate) bind_group_layouts: Vec<wgpu::BindGroupLayout>,
    #[cfg(feature = "gpu")]
    /// Bind group layout for dynamic-focus DAS (7 bindings: RF, output, params,
    /// apodization, element positions, focus delays, aperture masks).
    pub(crate) dynamic_focus_bind_group_layout: wgpu::BindGroupLayout,
    #[cfg(feature = "gpu")]
    /// Streaming data buffer
    pub(crate) streaming_buffer: Option<super::streaming::StreamingBuffer>,
    /// Performance metrics
    pub(crate) metrics: BeamformingMetrics,
}

impl BeamformingProcessor3D {
    /// Create new 3D beamforming processor with GPU initialization
    ///
    /// # GPU Setup
    /// 1. Request high-performance GPU adapter
    /// 2. Create logical device and command queue
    /// 3. Load WGSL compute shaders
    /// 4. Configure bind group layouts for buffers
    /// 5. Create compute pipelines
    /// 6. Initialize streaming buffer if enabled
    ///
    /// # Errors
    /// - Returns `ResourceUnavailable` if GPU adapter or device cannot be acquired
    /// - Returns `FeatureNotAvailable` if compiled without `gpu` feature
    pub async fn new(config: BeamformingConfig3D) -> KwaversResult<Self> {
        #[cfg(not(feature = "gpu"))]
        {
            let _ = config;
            Err(KwaversError::System(
                crate::core::error::SystemError::FeatureNotAvailable {
                    feature: "gpu".to_owned(),
                    reason:
                        "GPU acceleration required for 3D beamforming. Enable with --features gpu"
                            .to_owned(),
                },
            ))
        }

        #[cfg(feature = "gpu")]
        {
            // Initialize GPU instance
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });

            // Request high-performance GPU adapter
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .map_err(|_| {
                    KwaversError::System(crate::core::error::SystemError::ResourceUnavailable {
                        resource: "GPU adapter for 3D beamforming".to_string(),
                    })
                })?;

            // Create logical device and queue
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("3D Beamforming Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: wgpu::Trace::Off,
                })
                .await
                .map_err(|e| {
                    KwaversError::System(crate::core::error::SystemError::ResourceUnavailable {
                        resource: format!("GPU device for 3D beamforming: {}", e),
                    })
                })?;

            // Load WGSL compute shaders
            let delay_sum_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("3D Delay-and-Sum Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::BEAMFORMING_3D_SHADER.into()),
            });

            let dynamic_focus_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("3D Dynamic-Focus Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::DYNAMIC_FOCUS_3D_SHADER.into()),
            });

            // Helper: create a standard read-only storage entry
            let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            };
            // Helper: create a read-write storage entry
            let storage_rw = |binding: u32| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            };
            // Helper: create a uniform entry
            let uniform = |binding: u32| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            };

            // Static DAS bind group layout (5 bindings).
            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("3D Beamforming Bind Group Layout"),
                    entries: &[
                        storage_ro(0), // RF data
                        storage_rw(1), // output volume
                        uniform(2),    // params
                        storage_ro(3), // apodization weights
                        storage_ro(4), // element positions
                    ],
                });

            // Dynamic-focus bind group layout (7 bindings).
            let dynamic_focus_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Dynamic-Focus Bind Group Layout"),
                    entries: &[
                        storage_ro(0), // RF data
                        storage_rw(1), // output volume
                        uniform(2),    // DynamicFocusParams
                        storage_ro(3), // apodization weights
                        storage_ro(4), // element positions
                        storage_ro(5), // focus delays [zones × elements]
                        storage_ro(6), // aperture masks [zones × ⌈elements/32⌉]
                    ],
                });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("3D Beamforming Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

            let dynamic_focus_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Dynamic-Focus Pipeline Layout"),
                    bind_group_layouts: &[&dynamic_focus_bind_group_layout],
                    push_constant_ranges: &[],
                });

            // Static DAS compute pipeline.
            let delay_sum_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("3D Delay-and-Sum Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &delay_sum_shader,
                    entry_point: Some("delay_and_sum_main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

            // Dynamic-focus compute pipeline.
            let dynamic_focus_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("3D Dynamic-Focus Pipeline"),
                    layout: Some(&dynamic_focus_pipeline_layout),
                    module: &dynamic_focus_shader,
                    entry_point: Some("dynamic_focus_main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

            // Initialize streaming buffer if enabled
            let streaming_buffer = if config.enable_streaming {
                Some(super::streaming::StreamingBuffer::new(
                    config.streaming_buffer_size,
                    config.num_elements_3d.0 * config.num_elements_3d.1 * config.num_elements_3d.2,
                    1024, // samples per channel
                ))
            } else {
                None
            };

            Ok(Self {
                config,
                device,
                queue,
                delay_sum_pipeline,
                dynamic_focus_pipeline,
                bind_group_layouts: vec![bind_group_layout],
                dynamic_focus_bind_group_layout,
                streaming_buffer,
                metrics: BeamformingMetrics::default(),
            })
        }
    }

    /// Get current performance metrics
    #[must_use]
    pub fn metrics(&self) -> &BeamformingMetrics {
        &self.metrics
    }

    /// Create apodization weights for sidelobe reduction.
    ///
    /// Delegates to [`super::apodization::create_apodization_weights`].
    /// Called from the GPU delay-and-sum dispatch path in `processing/algorithms.rs`.
    #[cfg(feature = "gpu")]
    pub(super) fn create_apodization_weights(
        &self,
        window: &super::config::ApodizationWindow,
    ) -> ndarray::Array3<f32> {
        super::apodization::create_apodization_weights(self.config.num_elements_3d, window)
    }

    /// Execute delay-and-sum beamforming on GPU
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[cfg(feature = "gpu")]
    pub(super) fn delay_and_sum_gpu(
        &self,
        rf_data: &ndarray::Array4<f32>,
        dynamic_focusing: bool,
        apodization_window: &super::config::ApodizationWindow,
        apodization_weights: &ndarray::Array3<f32>,
    ) -> KwaversResult<ndarray::Array3<f32>> {
        let delay_sum = super::delay_sum::DelaySumGPU::new(
            &self.config,
            &self.device,
            &self.queue,
            &self.delay_sum_pipeline,
            &self.bind_group_layouts[0],
        );
        delay_sum.process(
            rf_data,
            dynamic_focusing,
            apodization_window,
            apodization_weights,
        )
    }

    /// Execute dynamic-focus delay-and-sum on GPU.
    ///
    /// Uses `dynamic_focus_3d.wgsl` with CPU-pre-computed delay tables; the
    /// GPU kernel applies depth-stratified focal zones and optional variable
    /// aperture.
    /// # Errors
    /// - Propagates GPU device errors.
    ///
    #[cfg(feature = "gpu")]
    pub(super) fn dynamic_focus_gpu(
        &self,
        rf_data: &ndarray::Array4<f32>,
        apodization_window: &super::config::ApodizationWindow,
        apodization_weights: &ndarray::Array3<f32>,
    ) -> KwaversResult<ndarray::Array3<f32>> {
        let df = super::delay_sum::DynamicFocusGPU {
            config: &self.config,
            device: &self.device,
            queue: &self.queue,
            pipeline: &self.dynamic_focus_pipeline,
            bind_group_layout: &self.dynamic_focus_bind_group_layout,
        };
        df.process(rf_data, apodization_window, apodization_weights)
    }

    /// Execute delay-and-sum subvolume processing on GPU
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[cfg(feature = "gpu")]
    pub(super) fn delay_and_sum_subvolume_gpu(
        &self,
        rf_data: &ndarray::Array4<f32>,
        dynamic_focusing: bool,
        apodization_window: &super::config::ApodizationWindow,
        apodization_weights: &ndarray::Array3<f32>,
        sub_volume_size: (usize, usize, usize),
    ) -> KwaversResult<ndarray::Array3<f32>> {
        let delay_sum = super::delay_sum::DelaySumGPU::new(
            &self.config,
            &self.device,
            &self.queue,
            &self.delay_sum_pipeline,
            &self.bind_group_layouts[0],
        );
        delay_sum.process_subvolume(
            rf_data,
            dynamic_focusing,
            apodization_window,
            apodization_weights,
            sub_volume_size,
        )
    }

    /// Calculate GPU memory usage.
    ///
    /// Delegates to [`super::metrics::calculate_gpu_memory_usage`].
    /// Called from `processing/mod.rs` after each volume reconstruction.
    #[cfg(feature = "gpu")]
    pub(super) fn calculate_gpu_memory_usage(&self) -> f64 {
        super::metrics::calculate_gpu_memory_usage(&self.config)
    }

    /// Calculate streaming buffer CPU memory usage (GPU build).
    ///
    /// Delegates to [`super::metrics::calculate_cpu_memory_usage`].
    #[cfg(feature = "gpu")]
    pub(super) fn calculate_cpu_memory_usage(&self) -> f64 {
        super::metrics::calculate_cpu_memory_usage(&self.streaming_buffer)
    }

    /// Calculate CPU memory usage (non-GPU build).
    ///
    /// Delegates to [`super::metrics::calculate_cpu_memory_usage`].
    /// Returns `0.0`: no streaming buffer exists in CPU-only builds.
    #[cfg(not(feature = "gpu"))]
    pub(super) fn calculate_cpu_memory_usage(&self) -> f64 {
        super::metrics::calculate_cpu_memory_usage(&None::<()>)
    }
}
