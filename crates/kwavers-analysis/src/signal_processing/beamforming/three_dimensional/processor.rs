//! GPU-Accelerated 3D Beamforming Processor
//!
//! Core processor structure that owns provider-generic beamforming state and
//! real-time streaming buffers for volumetric ultrasound beamforming.
//!
//! # Architecture
//! - Provider-owned GPU device and queue management
//! - Provider-owned kernel setup for delay-and-sum and dynamic focusing
//! - Streaming buffer management for real-time 4D imaging
//!
//! # References
//! - Jensen (1996) - Field: A Program for Simulating Ultrasound Systems
//! - Synnevåg et al. (2005) - Adaptive beamforming applied to medical ultrasound imaging

use super::config::{BeamformingConfig3D, BeamformingMetrics};
#[cfg(feature = "gpu")]
use super::provider::BeamformingGpuProvider;
#[cfg(not(feature = "gpu"))]
use kwavers_core::error::KwaversError;
use kwavers_core::error::KwaversResult;

/// Real-time 3D beamforming processor with optional GPU acceleration
#[derive(Debug)]
#[cfg(feature = "gpu")]
pub struct BeamformingProcessor3D<P>
where
    P: BeamformingGpuProvider,
{
    /// Configuration — used by both GPU and CPU execution paths.
    pub(crate) config: BeamformingConfig3D,
    /// Provider-owned GPU backend.
    pub(super) provider: P,
    /// Streaming data buffer
    pub(crate) streaming_buffer: Option<super::streaming::StreamingBuffer>,
    /// Performance metrics
    pub(crate) metrics: BeamformingMetrics,
}

/// CPU-only 3-D beamforming processor.
#[cfg(not(feature = "gpu"))]
#[derive(Debug)]
pub struct BeamformingProcessor3D {
    /// Configuration used by CPU execution paths.
    pub(crate) config: BeamformingConfig3D,
    /// Performance metrics.
    pub(crate) metrics: BeamformingMetrics,
}

#[cfg(feature = "gpu")]
impl<P> BeamformingProcessor3D<P>
where
    P: BeamformingGpuProvider,
{
    /// Construct a processor from an already-acquired GPU provider.
    ///
    /// # Errors
    ///
    /// Returns an error when the provider-specific processor state cannot be
    /// assembled.
    pub fn with_provider(config: BeamformingConfig3D, provider: P) -> KwaversResult<Self> {
        let streaming_buffer = if config.enable_streaming {
            Some(super::streaming::StreamingBuffer::new(
                config.streaming_buffer_size,
                config.num_elements_3d.0 * config.num_elements_3d.1 * config.num_elements_3d.2,
                1024,
            ))
        } else {
            None
        };

        Ok(Self {
            config,
            provider,
            streaming_buffer,
            metrics: BeamformingMetrics::default(),
        })
    }

    /// Get current performance metrics
    #[must_use]
    pub fn metrics(&self) -> &BeamformingMetrics {
        &self.metrics
    }

    /// Return the selected GPU provider.
    #[must_use]
    pub fn gpu_provider(&self) -> kwavers_solver::backend::traits::GpuProvider {
        self.provider.provider_kind()
    }

    /// Create apodization weights for sidelobe reduction.
    ///
    /// Delegates to [`super::apodization::create_apodization_weights`].
    /// Called from the GPU delay-and-sum dispatch path in `processing/algorithms.rs`.
    #[cfg(feature = "gpu")]
    pub(super) fn create_apodization_weights(
        &self,
        window: &super::config::Beamforming3dApodizationWindow,
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
        apodization_window: &super::config::Beamforming3dApodizationWindow,
        apodization_weights: &ndarray::Array3<f32>,
    ) -> KwaversResult<ndarray::Array3<f32>> {
        self.provider.process_delay_and_sum(
            &self.config,
            rf_data,
            dynamic_focusing,
            apodization_window,
            apodization_weights,
            None,
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
        apodization_window: &super::config::Beamforming3dApodizationWindow,
        apodization_weights: &ndarray::Array3<f32>,
    ) -> KwaversResult<ndarray::Array3<f32>> {
        self.provider.process_delay_and_sum(
            &self.config,
            rf_data,
            true,
            apodization_window,
            apodization_weights,
            None,
        )
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
        apodization_window: &super::config::Beamforming3dApodizationWindow,
        apodization_weights: &ndarray::Array3<f32>,
        sub_volume_size: (usize, usize, usize),
    ) -> KwaversResult<ndarray::Array3<f32>> {
        self.provider.process_delay_and_sum(
            &self.config,
            rf_data,
            dynamic_focusing,
            apodization_window,
            apodization_weights,
            Some(sub_volume_size),
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
}

#[cfg(not(feature = "gpu"))]
impl BeamformingProcessor3D {
    /// Return the CPU-only build's GPU construction error.
    ///
    /// # Errors
    ///
    /// Returns `FeatureNotAvailable` because this processor currently exposes
    /// GPU-backed construction only; CPU algorithm dispatch is tested through
    /// the CPU kernels directly.
    pub fn new_wgpu(config: BeamformingConfig3D) -> KwaversResult<Self> {
        let _ = config;
        Err(KwaversError::System(
            kwavers_core::error::SystemError::FeatureNotAvailable {
                feature: "gpu".to_owned(),
                reason: "GPU acceleration requires constructing BeamformingProcessor3D with a provider from kwavers-gpu"
                    .to_owned(),
            },
        ))
    }

    /// Get current performance metrics.
    #[must_use]
    pub fn metrics(&self) -> &BeamformingMetrics {
        &self.metrics
    }

    /// Calculate CPU memory usage (non-GPU build).
    ///
    /// Delegates to [`super::metrics::calculate_cpu_memory_usage`].
    /// Returns `0.0`: no streaming buffer exists in CPU-only builds.
    pub(super) fn calculate_cpu_memory_usage(&self) -> f64 {
        super::metrics::calculate_cpu_memory_usage(&None::<()>)
    }
}
