use crate::gpu::memory::GpuMemoryPoolType;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::fft::{fft_1d_complex_slice_inplace, ifft_1d_complex_slice_inplace, Complex64};
use leto::{Array3 as LetoArray3, Array4 as LetoArray4};
use log::{debug, info, warn};
use moirai_parallel::{for_each_chunk_mut_enumerated_with, Adaptive};
use std::time::Instant;

use super::HILBERT_SPECTRUM;
use super::{PipelineState, PipelineStats, RealtimeImagingPipeline, RealtimePipelineConfig};

impl RealtimeImagingPipeline {
    /// New.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(config: RealtimePipelineConfig) -> KwaversResult<Self> {
        use crate::gpu::memory::UnifiedMemoryManager;
        use std::collections::VecDeque;
        use std::sync::{Arc, Mutex};

        let input_buffer = Arc::new(Mutex::new(VecDeque::with_capacity(config.buffer_size)));
        let output_buffer = Arc::new(Mutex::new(VecDeque::with_capacity(config.buffer_size)));

        let gpu_memory = if config.gpu_accelerated {
            Some(UnifiedMemoryManager::new())
        } else {
            None
        };

        Ok(Self {
            config,
            input_buffer,
            output_buffer,
            gpu_memory,
            stats: PipelineStats::default(),
            state: PipelineState::Stopped,
        })
    }
    /// Start.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn start(&mut self) -> KwaversResult<()> {
        info!("Starting real-time imaging pipeline...");
        info!(
            "Target FPS: {:.1}, Max latency: {:.1} ms",
            self.config.target_fps, self.config.max_latency_ms
        );

        self.state = PipelineState::Starting;

        if let Some(ref mut gpu_mem) = self.gpu_memory {
            gpu_mem.allocate(0, GpuMemoryPoolType::Temporary, 1024 * 1024)?;
            gpu_mem.allocate(0, GpuMemoryPoolType::Persistent, 512 * 1024)?;
        }

        self.state = PipelineState::Running;
        info!("Pipeline started successfully");
        Ok(())
    }
    /// Stop.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn stop(&mut self) -> KwaversResult<()> {
        debug!("Stopping real-time imaging pipeline...");
        self.state = PipelineState::Stopping;

        {
            let mut input = self.input_buffer.lock().unwrap_or_else(|e| e.into_inner());
            input.clear();
        }
        {
            let mut output = self.output_buffer.lock().unwrap_or_else(|e| e.into_inner());
            output.clear();
        }

        self.state = PipelineState::Stopped;
        info!("Pipeline stopped");
        Ok(())
    }
    /// Submit rf data.
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn submit_rf_data(&mut self, rf_data: LetoArray4<f32>) -> KwaversResult<()> {
        if self.state != PipelineState::Running {
            return Err(KwaversError::InvalidInput(
                "Pipeline is not running".to_string(),
            ));
        }

        let mut buffer = self.input_buffer.lock().unwrap_or_else(|e| e.into_inner());

        if buffer.len() >= self.config.buffer_size {
            self.stats.dropped_frames += 1;
            warn!("Input buffer full, dropping frame");
            return Ok(());
        }

        buffer.push_back(rf_data);
        Ok(())
    }
    /// Get processed frame.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn get_processed_frame(&mut self) -> Option<LetoArray3<f32>> {
        let mut buffer = self.output_buffer.lock().unwrap_or_else(|e| e.into_inner());
        buffer.pop_front()
    }
    /// Process pipeline.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn process_pipeline(&mut self) -> KwaversResult<()> {
        if self.state != PipelineState::Running {
            return Ok(());
        }

        let start_time = Instant::now();

        let rf_data = {
            let mut buffer = self.input_buffer.lock().unwrap_or_else(|e| e.into_inner());
            buffer.pop_front()
        };

        if let Some(data) = rf_data {
            let processed_frame = self.process_frame(&data)?;

            {
                let mut output = self.output_buffer.lock().unwrap_or_else(|e| e.into_inner());
                if output.len() < self.config.buffer_size {
                    output.push_back(processed_frame);
                } else {
                    self.stats.dropped_frames += 1;
                }
            }

            let processing_time = start_time.elapsed();
            self.stats.frames_processed += 1;
            self.stats.total_processing_time += processing_time;

            if self.stats.frames_processed > 0 {
                self.stats.average_latency =
                    self.stats.total_processing_time / self.stats.frames_processed as u32;
            }

            let latency_ms = processing_time.as_secs_f64() * 1000.0;
            if latency_ms > self.config.max_latency_ms {
                warn!(
                    "Processing latency {:.1}ms exceeds limit {:.1}ms",
                    latency_ms, self.config.max_latency_ms
                );
            }
        }

        Ok(())
    }

    fn process_frame(&mut self, rf_data: &LetoArray4<f32>) -> KwaversResult<LetoArray3<f32>> {
        let beamformed = self.beamform(rf_data)?;
        let envelope = self.envelope_detection(&beamformed)?;
        let compressed = self.log_compression(&envelope)?;
        self.scan_conversion(&compressed)
    }

    fn beamform(&self, rf_data: &LetoArray4<f32>) -> KwaversResult<LetoArray3<f32>> {
        let [tx_count, rx_count, samples, frames] = rf_data.shape();
        let mut beamformed = LetoArray3::zeros([rx_count, samples, frames]);

        for tx in 0..tx_count {
            for rx in 0..rx_count {
                for sample in 0..samples {
                    for frame in 0..frames {
                        beamformed[[rx, sample, frame]] += rf_data[[tx, rx, sample, frame]];
                    }
                }
            }
        }

        Ok(beamformed)
    }

    fn envelope_detection(
        &mut self,
        beamformed: &LetoArray3<f32>,
    ) -> KwaversResult<LetoArray3<f32>> {
        let [rx_count, samples, frames] = beamformed.shape();
        let mut envelope = LetoArray3::zeros([rx_count, samples, frames]);
        let rx_plane_len = samples * frames;

        match (
            envelope.as_slice_memory_order_mut(),
            beamformed.as_slice_memory_order(),
        ) {
            (Some(envelope_values), Some(beamformed_values)) => {
                for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
                    envelope_values,
                    rx_plane_len,
                    |rx, envelope_rx| {
                        let start = rx * rx_plane_len;
                        let beamformed_rx = &beamformed_values[start..start + envelope_rx.len()];
                        compute_envelope_plane(samples, frames, beamformed_rx, envelope_rx);
                    },
                );
            }
            _ => {
                for rx in 0..rx_count {
                    compute_envelope_plane_indexed(
                        samples,
                        frames,
                        |sample, frame| beamformed[[rx, sample, frame]],
                        |sample, frame, value| envelope[[rx, sample, frame]] = value,
                    );
                }
            }
        }

        Ok(envelope)
    }

    fn log_compression(&self, envelope: &LetoArray3<f32>) -> KwaversResult<LetoArray3<f32>> {
        let mut compressed = LetoArray3::zeros(envelope.shape());
        let dynamic_range_db = 60.0_f64;
        let compression_factor: f32 = (dynamic_range_db / 20.0) as f32;

        for i in 0..envelope.shape()[0] {
            for j in 0..envelope.shape()[1] {
                for k in 0..envelope.shape()[2] {
                    let value = envelope[[i, j, k]].max(1e-10) as f64;
                    let log_value = value.ln() / (compression_factor as f64).ln();
                    compressed[[i, j, k]] = log_value.clamp(0.0, 1.0) as f32;
                }
            }
        }

        Ok(compressed)
    }

    fn scan_conversion(&self, compressed: &LetoArray3<f32>) -> KwaversResult<LetoArray3<f32>> {
        Ok(compressed.clone())
    }

    pub fn statistics(&self) -> &super::PipelineStats {
        &self.stats
    }

    pub fn state(&self) -> PipelineState {
        self.state
    }

    pub fn is_ready(&self) -> bool {
        let buffer = self.input_buffer.lock().unwrap_or_else(|e| e.into_inner());
        buffer.len() < self.config.buffer_size
    }

    pub fn buffer_utilization(&self) -> (f64, f64) {
        let input_len = self
            .input_buffer
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len();
        let output_len = self
            .output_buffer
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len();

        (
            input_len as f64 / self.config.buffer_size as f64,
            output_len as f64 / self.config.buffer_size as f64,
        )
    }

    pub fn set_adaptive_processing(&mut self, enabled: bool) {
        self.config.adaptive_processing = enabled;
    }

    pub fn set_target_fps(&mut self, fps: f64) {
        self.config.target_fps = fps;
    }
}

fn compute_envelope_plane(samples: usize, frames: usize, beamformed: &[f32], envelope: &mut [f32]) {
    compute_envelope_plane_indexed(
        samples,
        frames,
        |sample, frame| beamformed[sample * frames + frame],
        |sample, frame, value| envelope[sample * frames + frame] = value,
    );
}

fn compute_envelope_plane_indexed<Read, Write>(
    samples: usize,
    frames: usize,
    read: Read,
    mut write: Write,
) where
    Read: Fn(usize, usize) -> f32,
    Write: FnMut(usize, usize, f32),
{
    if samples == 0 {
        return;
    }

    HILBERT_SPECTRUM.with(|spectrum_cell| {
        let mut spectrum = spectrum_cell.borrow_mut();
        if spectrum.len() != samples {
            spectrum.resize(samples, Complex64::default());
        }

        for frame in 0..frames {
            for sample in 0..samples {
                spectrum[sample] = Complex64::new(read(sample, frame) as f64, 0.0);
            }

            fft_1d_complex_slice_inplace(spectrum.as_mut_slice());

            if samples > 1 {
                if samples.is_multiple_of(2) {
                    for coeff in spectrum.iter_mut().take(samples / 2).skip(1) {
                        *coeff *= 2.0;
                    }
                    for coeff in spectrum.iter_mut().skip(samples / 2 + 1) {
                        *coeff = Complex64::default();
                    }
                } else {
                    for coeff in spectrum.iter_mut().take(samples / 2 + 1).skip(1) {
                        *coeff *= 2.0;
                    }
                    for coeff in spectrum.iter_mut().skip(samples / 2 + 1) {
                        *coeff = Complex64::default();
                    }
                }
            }

            ifft_1d_complex_slice_inplace(spectrum.as_mut_slice());

            let norm = 1.0 / samples as f64;
            for sample in 0..samples {
                write(sample, frame, (spectrum[sample] * norm).norm() as f32);
            }
        }
    });
}