//! Processing entry points for 3D beamforming operations.
//!
//! This module provides the main processing interface for volumetric ultrasound
//! beamforming, including single-volume and real-time streaming modes.

mod algorithms;
#[cfg(test)]
mod tests;
mod validation;

use kwavers_core::error::{KwaversError, KwaversResult};
use leto::{
    Array3,
    Array4,
};

use super::config::BeamformingAlgorithm3D;
use super::processor::BeamformingProcessor3D;
#[cfg(feature = "gpu")]
use super::provider::BeamformingGpuProvider;
use super::SaftProcessor;

#[cfg(feature = "gpu")]
impl<P> BeamformingProcessor3D<P>
where
    P: BeamformingGpuProvider,
{
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
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[cfg(feature = "gpu")]
    pub fn process_volume(
        &mut self,
        rf_data: &Array4<f32>,
        algorithm: &BeamformingAlgorithm3D,
    ) -> KwaversResult<Array3<f32>> {
        let start_time = std::time::Instant::now();
        self.validate_input(rf_data)?;

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
                let saft_processor = SaftProcessor::from_algorithm(algorithm, self.config.clone())?;
                saft_processor.reconstruct_volume(rf_data)?
            }
        };

        let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;
        self.metrics.processing_time_ms = processing_time;
        self.metrics.reconstruction_rate = 1000.0 / processing_time;
        self.metrics.gpu_memory_mb = self.calculate_gpu_memory_usage();
        self.metrics.cpu_memory_mb = self.calculate_cpu_memory_usage();

        Ok(volume)
    }
}

#[cfg(not(feature = "gpu"))]
impl BeamformingProcessor3D {
    /// CPU 3D beamforming dispatcher — active when the `gpu` feature is absent.
    ///
    /// Dispatches to the analytically specified CPU kernels in `super::cpu`:
    /// - `DelayAndSum` → `cpu::delay_and_sum_cpu`
    /// - `MVDR3D`     → `cpu::mvdr_cpu`
    /// - `SAFT3D`     → [`SaftProcessor::reconstruct_volume`] (pure-CPU, no GPU dependency)
    ///
    /// Dynamic focusing and sub-volume chunking flags are accepted but forwarded
    /// only to kernels that implement them; the CPU DAS kernel processes the
    /// full volume in a single Rayon-parallel pass.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[cfg(not(feature = "gpu"))]
    pub fn process_volume(
        &mut self,
        rf_data: &Array4<f32>,
        algorithm: &BeamformingAlgorithm3D,
    ) -> KwaversResult<Array3<f32>> {
        let start_time = std::time::Instant::now();
        self.validate_input(rf_data)?;

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
                let saft = SaftProcessor::from_algorithm(algorithm, self.config.clone())?;
                saft.reconstruct_volume(rf_data)?
            }
        };

        let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;
        self.metrics.processing_time_ms = processing_time;
        self.metrics.reconstruction_rate = 1000.0 / processing_time.max(f64::EPSILON);
        self.metrics.cpu_memory_mb = self.calculate_cpu_memory_usage();

        Ok(volume)
    }
}

#[cfg(feature = "gpu")]
impl<P> BeamformingProcessor3D<P>
where
    P: BeamformingGpuProvider,
{
    /// Process streaming data for real-time 4D imaging.
    ///
    /// Accumulates frames into a buffer and processes complete volumes when ready.
    ///
    /// # Returns
    /// - `Some(volume)` when a complete volume is ready
    /// - `None` if buffer is still accumulating frames
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[cfg(feature = "gpu")]
    pub fn process_streaming(
        &mut self,
        rf_frame: &Array3<f32>,
        algorithm: &BeamformingAlgorithm3D,
    ) -> KwaversResult<Option<Array3<f32>>> {
        if self.streaming_buffer.is_none() {
            return Err(KwaversError::InvalidInput(
                "Streaming not enabled in configuration".to_string(),
            ));
        }

        if !self
            .streaming_buffer
            .as_mut()
            .unwrap()
            .add_frame(rf_frame)?
        {
            return Ok(None);
        }

        let rf_data = self
            .streaming_buffer
            .as_ref()
            .unwrap()
            .get_volume_data()
            .clone();
        self.process_volume(&rf_data, algorithm).map(Some)
    }
}

#[cfg(not(feature = "gpu"))]
impl BeamformingProcessor3D {
    /// CPU streaming entry point — wraps a single RF frame as a 1-frame volume
    /// and delegates to [`Self::process_volume`].
    ///
    /// Without GPU streaming buffers the processor has no accumulation state,
    /// so every frame produces a complete reconstructed volume immediately.
    /// This is semantically equivalent to single-frame compounding.
    ///
    /// # Frame shape contract
    /// `rf_frame` must be `[channels, samples, 1]` — the same trailing-1 layout
    /// used by the GPU streaming buffer.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[cfg(not(feature = "gpu"))]
    pub fn process_streaming(
        &mut self,
        rf_frame: &Array3<f32>,
        algorithm: &BeamformingAlgorithm3D,
    ) -> KwaversResult<Option<Array3<f32>>> {
        let (channels, samples, _trailing) = rf_frame.dim();
        let rf_data = rf_frame
            .view()
            .into_shape_with_order((1, channels, samples, 1_usize))
            .map_err(|e| {
                KwaversError::InvalidInput(format!(
                    "streaming frame reshape [channels={channels}, samples={samples}, 1]: {e}"
                ))
            })?
            .to_owned();
        self.process_volume(&rf_data, algorithm).map(Some)
    }
}
