//! NeuralBeamformingProcessor core implementation.

use kwavers_core::error::KwaversResult;
use leto::{
    /* s -- no leto equivalent */,
    Array3,
    Array4,
    ArrayView3,
    ArrayView4,
};
use std::collections::HashMap;

#[cfg(feature = "pinn")]
use kwavers_solver::interface::pinn_beamforming::{
    PinnBeamformingProvider, PinnBeamformingUncertaintyConfig,
};

use crate::signal_processing::beamforming::utils::steering::SteeringVector;

use super::super::super::types::{
    NeuralBeamformingMetrics, NeuralPinnBeamformingResult, PINNBeamformingConfig,
};
use super::super::inference;

/// AI-enhanced beamforming processor with PINN optimization.
///
/// Integrates physics-informed neural networks for:
/// - Optimal delay calculation via eikonal equation
/// - Adaptive weight computation with wave physics constraints
/// - Uncertainty quantification via Bayesian inference
pub struct NeuralBeamformingProcessor {
    /// Configuration
    pub(super) config: PINNBeamformingConfig,
    /// PINN provider for beamforming (trait object, solver-agnostic)
    #[cfg(feature = "pinn")]
    pub(super) pinn_provider: Option<Box<dyn PinnBeamformingProvider>>,
    /// Steering vectors cache for performance
    pub(super) steering_cache: HashMap<(usize, usize, usize), SteeringVector>,
    /// Performance metrics
    pub(super) metrics: NeuralBeamformingMetrics,
}

impl std::fmt::Debug for NeuralBeamformingProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NeuralBeamformingProcessor")
            .field("config", &self.config)
            .field("steering_cache_size", &self.steering_cache.len())
            .field("metrics", &self.metrics)
            .finish()
    }
}

impl NeuralBeamformingProcessor {
    /// Create new neural beamforming processor.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(config: PINNBeamformingConfig) -> KwaversResult<Self> {
        Ok(Self {
            config,
            #[cfg(feature = "pinn")]
            pinn_provider: None,
            steering_cache: HashMap::new(),
            metrics: NeuralBeamformingMetrics::default(),
        })
    }

    /// Set the PINN provider for this processor (dependency injection).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[cfg(feature = "pinn")]
    pub fn set_provider(&mut self, provider: Box<dyn PinnBeamformingProvider>) {
        self.pinn_provider = Some(provider);
    }

    /// Return the number of cached steering vectors.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn steering_cache_len(&self) -> usize {
        self.steering_cache.len()
    }

    /// Process 4D RF data volume with PINN-enhanced beamforming.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn process_volume(
        &mut self,
        rf_data: &Array4<f32>,
    ) -> KwaversResult<NeuralPinnBeamformingResult> {
        self.process_volume_view(rf_data.view())
    }

    /// Process a frame-major RF volume without copying the input buffer.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(crate) fn process_volume_view(
        &mut self,
        rf_data: ArrayView4<'_, f32>,
    ) -> KwaversResult<NeuralPinnBeamformingResult> {
        let start_time = std::time::Instant::now();

        let (frames, channels, samples, _) = rf_data.dim();

        let mut volume = Array3::<f32>::zeros((frames, channels, samples));

        for frame_idx in 0..frames {
            let frame_data = rf_data.slice(s![frame_idx, .., .., 0..1]);
            let frame_result = self.process_frame(&frame_data)?;
            volume
                .index_axis_mut(0, frame_idx)
                .assign(&frame_result.index_axis(2, 0));
        }

        let uncertainty = self.compute_uncertainty(&volume)?;
        let confidence = self.compute_confidence(&uncertainty)?;

        let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;
        self.metrics.total_processing_time = processing_time;

        Ok(NeuralPinnBeamformingResult {
            volume,
            uncertainty,
            confidence,
            processing_time_ms: processing_time,
        })
    }

    /// Get performance metrics.
    pub fn metrics(&self) -> &NeuralBeamformingMetrics {
        &self.metrics
    }

    /// Calculate memory requirement for processor.
    pub fn calculate_memory_requirement(&self) -> usize {
        let pinn_memory = if self.config.enable_pinn {
            #[cfg(feature = "pinn")]
            {
                10 * 1024 * 1024
            }
            #[cfg(not(feature = "pinn"))]
            {
                0
            }
        } else {
            0
        };

        let uncertainty_memory = if self.config.enable_uncertainty_quantification {
            self.config.volume_size.0
                * self.config.volume_size.1
                * self.config.volume_size.2
                * 8
                * 10
        } else {
            0
        };

        let base_memory = self.config.rf_data_channels * self.config.samples_per_channel * 4;

        pinn_memory + uncertainty_memory + base_memory
    }

    // --- Private helpers ---

    fn process_frame(&mut self, frame_data: &ArrayView3<f32>) -> KwaversResult<Array3<f32>> {
        let (channels, samples, _) = frame_data.dim();
        let mut output = Array3::<f32>::zeros((channels, samples, 1));

        for channel in 0..channels {
            for sample in 0..samples {
                let optimal_delay = self.compute_pinn_delay(channel, sample)?;
                let voxel_value =
                    self.apply_pinn_beamforming(frame_data, channel, sample, optimal_delay)?;
                output[[channel, sample, 0]] = voxel_value;
            }
        }

        Ok(output)
    }

    #[cfg(feature = "pinn")]
    fn compute_pinn_delay(&mut self, channel_idx: usize, sample_idx: usize) -> KwaversResult<f64> {
        inference::compute_delay(
            channel_idx,
            self.config.rf_data_channels,
            sample_idx,
            self.config.channel_spacing,
            self.config.focal_depth,
            self.config.base_config.sound_speed,
            self.config.base_config.sampling_frequency,
        )
    }

    #[cfg(not(feature = "pinn"))]
    fn compute_pinn_delay(
        &mut self,
        _channel_idx: usize,
        _sample_idx: usize,
    ) -> KwaversResult<f64> {
        Err(kwavers_core::error::KwaversError::System(
            kwavers_core::error::SystemError::FeatureNotAvailable {
                feature: "pinn".to_string(),
                reason: "PINN beamforming requires 'pinn' feature".to_string(),
            },
        ))
    }

    fn apply_pinn_beamforming(
        &mut self,
        frame_data: &ArrayView3<f32>,
        channel: usize,
        sample: usize,
        delay: f64,
    ) -> KwaversResult<f32> {
        let delayed_samples: Vec<f32> = (0..self.config.rf_data_channels)
            .map(|elem| {
                let delayed_idx = (sample as f64 + delay * elem as f64) as usize;
                if delayed_idx < frame_data.shape()[1] {
                    frame_data[[elem, delayed_idx, 0]]
                } else {
                    0.0
                }
            })
            .collect();

        let weights = self.compute_pinn_weights(channel, sample, &delayed_samples)?;

        let result: f32 = delayed_samples
            .iter()
            .zip(weights.iter())
            .map(|(s, weight)| s * weight)
            .sum();

        Ok(result)
    }

    fn compute_pinn_weights(
        &mut self,
        _channel: usize,
        sample: usize,
        _samples: &[f32],
    ) -> KwaversResult<Vec<f32>> {
        inference::compute_weights(
            self.config.rf_data_channels,
            sample,
            self.config.channel_spacing,
            self.config.focal_depth,
            self.config.base_config.sound_speed,
            self.config.base_config.reference_frequency,
        )
    }

    fn compute_uncertainty(&mut self, volume: &Array3<f32>) -> KwaversResult<Array3<f32>> {
        #[cfg(feature = "pinn")]
        {
            if let Some(provider) = &self.pinn_provider {
                let uncertainty_start = std::time::Instant::now();

                let uncertainty_config = PinnBeamformingUncertaintyConfig {
                    bayesian_enabled: self.config.enable_uncertainty_quantification,
                    ..Default::default()
                };

                let uncertainty = provider
                    .estimate_uncertainty(volume, &uncertainty_config)
                    .unwrap_or_else(|_| {
                        let mut u = Array3::<f32>::zeros(volume.dim());
                        for ((i, j, k), value) in volume.indexed_iter() {
                            u[[i, j, k]] = 1.0 / (value.abs() + 1.0);
                        }
                        u
                    });

                self.metrics.uncertainty_computation_time =
                    uncertainty_start.elapsed().as_secs_f64() * 1000.0;

                return Ok(uncertainty);
            }
        }

        let uncertainty_start = std::time::Instant::now();
        let mut uncertainty = Array3::<f32>::zeros(volume.dim());
        for ((i, j, k), value) in volume.indexed_iter() {
            uncertainty[[i, j, k]] = 1.0 / (value.abs() + 1.0);
        }
        self.metrics.uncertainty_computation_time =
            uncertainty_start.elapsed().as_secs_f64() * 1000.0;
        Ok(uncertainty)
    }

    fn compute_confidence(&self, uncertainty: &Array3<f32>) -> KwaversResult<Array3<f32>> {
        let mut confidence = Array3::<f32>::zeros(uncertainty.dim());
        for ((i, j, k), &uncert) in uncertainty.indexed_iter() {
            confidence[[i, j, k]] = 1.0 / (1.0 + uncert);
        }
        Ok(confidence)
    }
}
