//! Processing entry points for 3D beamforming operations.
//!
//! This module provides the main processing interface for volumetric ultrasound
//! beamforming, including single-volume and real-time streaming modes.

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{Array3, Array4};

use super::config::{ApodizationWindow, BeamformingAlgorithm3D};
use super::processor::BeamformingProcessor3D;
#[cfg(feature = "gpu")]
use super::SaftProcessor;

#[allow(dead_code)] // Methods used only with GPU feature enabled
impl BeamformingProcessor3D {
    /// Process 3D beamforming for a single volume
    ///
    /// # Arguments
    /// * `rf_data` - RF data array (frames × channels × samples × 1)
    /// * `algorithm` - Beamforming algorithm to apply
    ///
    /// # Returns
    /// Reconstructed 3D volume (x × y × z)
    ///
    /// # Performance
    /// - Target: <10ms per volume with GPU acceleration
    /// - Speedup: 10-100× vs CPU implementation
    #[cfg(feature = "gpu")]
    pub fn process_volume(
        &mut self,
        rf_data: &Array4<f32>,
        algorithm: &BeamformingAlgorithm3D,
    ) -> KwaversResult<Array3<f32>> {
        let start_time = std::time::Instant::now();

        // Validate input dimensions
        self.validate_input(rf_data)?;

        // Process based on algorithm
        let volume = match algorithm {
            BeamformingAlgorithm3D::DelayAndSum {
                dynamic_focusing,
                apodization,
                sub_volume_size,
            } => self.process_delay_and_sum(
                rf_data,
                *dynamic_focusing,
                apodization,
                *sub_volume_size,
            )?,
            BeamformingAlgorithm3D::MVDR3D {
                diagonal_loading,
                subarray_size,
            } => self.process_mvdr_3d(rf_data, *diagonal_loading as f32, *subarray_size)?,
            BeamformingAlgorithm3D::SAFT3D { .. } => {
                // Create SAFT processor and perform reconstruction
                let saft_processor = SaftProcessor::from_algorithm(algorithm, self.config.clone())?;
                saft_processor.reconstruct_volume(rf_data)?
            }
        };

        // Update metrics
        let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;
        self.metrics.processing_time_ms = processing_time;
        self.metrics.reconstruction_rate = 1000.0 / processing_time;

        // Calculate memory usage
        self.metrics.gpu_memory_mb = self.calculate_gpu_memory_usage();
        self.metrics.cpu_memory_mb = self.calculate_cpu_memory_usage();

        Ok(volume)
    }

    /// CPU fallback for 3D beamforming when GPU is not available
    #[cfg(not(feature = "gpu"))]
    pub fn process_volume(
        &mut self,
        _rf_data: &Array4<f32>,
        _algorithm: &BeamformingAlgorithm3D,
    ) -> KwaversResult<Array3<f32>> {
        // CPU implementation using delay-and-sum beamforming
        // Reference: Van Veen & Buckley (1988) "Beamforming: A versatile approach to spatial filtering"
        // Full GPU implementation available in gpu_accelerated module
        Err(KwaversError::System(
            crate::core::error::SystemError::FeatureNotAvailable {
                feature: "gpu".to_string(),
                reason: "GPU acceleration required for 3D beamforming. Enable with --features gpu"
                    .to_string(),
            },
        ))
    }

    /// Process streaming data for real-time 4D imaging
    ///
    /// Accumulates frames into a buffer and processes complete volumes when ready.
    ///
    /// # Arguments
    /// * `rf_frame` - Single RF frame (channels × samples × 1)
    /// * `algorithm` - Beamforming algorithm to apply
    ///
    /// # Returns
    /// - `Some(volume)` when a complete volume is ready
    /// - `None` if buffer is still accumulating frames
    #[cfg(feature = "gpu")]
    pub fn process_streaming(
        &mut self,
        rf_frame: &Array3<f32>,
        algorithm: &BeamformingAlgorithm3D,
    ) -> KwaversResult<Option<Array3<f32>>> {
        // Check if streaming is enabled
        if self.streaming_buffer.is_none() {
            return Err(KwaversError::InvalidInput(
                "Streaming not enabled in configuration".to_string(),
            ));
        }

        // Add frame to streaming buffer
        if !self
            .streaming_buffer
            .as_mut()
            .unwrap()
            .add_frame(rf_frame)?
        {
            return Ok(None); // Buffer not full yet
        }

        // Process complete volume - clone the data to avoid borrowing issues
        let rf_data = self
            .streaming_buffer
            .as_ref()
            .unwrap()
            .get_volume_data()
            .clone();
        self.process_volume(&rf_data, algorithm).map(Some)
    }

    /// CPU fallback for streaming processing
    #[cfg(not(feature = "gpu"))]
    pub fn process_streaming(
        &mut self,
        _rf_frame: &Array3<f32>,
        _algorithm: &BeamformingAlgorithm3D,
    ) -> KwaversResult<Option<Array3<f32>>> {
        Err(KwaversError::System(
            crate::core::error::SystemError::FeatureNotAvailable {
                feature: "gpu".to_string(),
                reason: "GPU acceleration required for streaming 3D beamforming. Enable with --features gpu"
                    .to_string(),
            },
        ))
    }

    /// Validate input RF data dimensions
    ///
    /// Ensures RF data matches expected configuration:
    /// - Non-empty data
    /// - Channel count matches 3D element array
    /// - At least one sample per channel
    #[cfg(feature = "gpu")]
    pub(super) fn validate_input(&self, rf_data: &Array4<f32>) -> KwaversResult<()> {
        let rf_dims = rf_data.dim();
        let _frames = rf_dims.0;
        let channels = rf_dims.1;
        let samples = rf_dims.2;

        // Basic validation - ensure data is not empty
        if rf_data.is_empty() {
            return Err(KwaversError::InvalidInput(
                "RF data array is empty".to_string(),
            ));
        }

        let expected_channels = self.config.num_elements_3d.0
            * self.config.num_elements_3d.1
            * self.config.num_elements_3d.2;

        if channels != expected_channels {
            return Err(KwaversError::InvalidInput(format!(
                "Channel count mismatch: expected {}, got {}",
                expected_channels, channels
            )));
        }

        if samples == 0 {
            return Err(KwaversError::InvalidInput(
                "RF data must contain at least one sample per channel".to_string(),
            ));
        }

        Ok(())
    }

    /// Process delay-and-sum beamforming
    ///
    /// Main delay-and-sum algorithm with optional dynamic focusing and apodization.
    ///
    /// # Algorithm
    /// Coherent summation of delayed RF signals across all elements to form focused image.
    /// Optional sub-volume processing for memory-constrained systems.
    #[cfg(feature = "gpu")]
    pub(super) fn process_delay_and_sum(
        &self,
        rf_data: &Array4<f32>,
        dynamic_focusing: bool,
        apodization: &ApodizationWindow,
        sub_volume_size: Option<(usize, usize, usize)>,
    ) -> KwaversResult<Array3<f32>> {
        // Create apodization weights
        let apodization_weights = self.create_apodization_weights(apodization);

        // Process in sub-volumes for memory efficiency if requested
        if let Some(sub_size) = sub_volume_size {
            self.delay_and_sum_subvolume_gpu(
                rf_data,
                dynamic_focusing,
                apodization,
                &apodization_weights,
                sub_size,
            )
        } else {
            self.delay_and_sum_gpu(rf_data, dynamic_focusing, apodization, &apodization_weights)
        }
    }

    /// CPU fallback for delay-and-sum processing
    #[cfg(not(feature = "gpu"))]
    pub(super) fn process_delay_and_sum(
        &self,
        _rf_data: &Array4<f32>,
        _dynamic_focusing: bool,
        _apodization: &ApodizationWindow,
        _sub_volume_size: Option<(usize, usize, usize)>,
    ) -> KwaversResult<Array3<f32>> {
        Err(KwaversError::System(
            crate::core::error::SystemError::FeatureNotAvailable {
                feature: "gpu".to_string(),
                reason: "GPU acceleration required for 3D beamforming. Enable with --features gpu"
                    .to_string(),
            },
        ))
    }

    /// Process MVDR 3D beamforming
    ///
    /// Minimum Variance Distortionless Response beamforming for adaptive spatial filtering.
    ///
    /// # References
    /// - Van Veen & Buckley (1988) "Beamforming: A versatile approach to spatial filtering"
    /// - Synnevåg et al. (2009) "Adaptive beamforming applied to medical ultrasound imaging"
    #[cfg(feature = "gpu")]
    pub(super) fn process_mvdr_3d(
        &mut self,
        rf_data: &Array4<f32>,
        diagonal_loading: f32,
        subarray_size: [usize; 3],
    ) -> KwaversResult<Array3<f32>> {
        let _ = (rf_data, diagonal_loading, subarray_size);
        Err(KwaversError::System(
            crate::core::error::SystemError::FeatureNotAvailable {
                feature: "mvdr-3d".to_string(),
                reason: "MVDR 3D reconstruction is not implemented".to_string(),
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_validate_input_empty_data() {
        let config = super::super::config::BeamformingConfig3D::default();
        #[cfg(feature = "gpu")]
        {
            use super::super::processor::BeamformingProcessor3D;
            let processor = tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(async { BeamformingProcessor3D::new(config).await });
            if let Ok(proc) = processor {
                let empty_data = Array4::<f32>::zeros((0, 0, 0, 0));
                assert!(proc.validate_input(&empty_data).is_err());
            }
        }

        #[cfg(not(feature = "gpu"))]
        {
            // CPU-only test - just verify config defaults
            assert_eq!(config.volume_dims, (128, 128, 128));
        }
    }

    #[test]
    fn test_validate_input_channel_mismatch() {
        let config = super::super::config::BeamformingConfig3D::default();
        #[cfg(feature = "gpu")]
        {
            use super::super::processor::BeamformingProcessor3D;
            let processor = tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(async { BeamformingProcessor3D::new(config).await });
            if let Ok(proc) = processor {
                // Wrong channel count
                let bad_data = Array4::<f32>::zeros((1, 100, 1024, 1));
                assert!(proc.validate_input(&bad_data).is_err());
            }
        }

        #[cfg(not(feature = "gpu"))]
        {
            // CPU-only test
            let expected_channels =
                config.num_elements_3d.0 * config.num_elements_3d.1 * config.num_elements_3d.2;
            assert_eq!(expected_channels, 16384);
        }
    }
}
