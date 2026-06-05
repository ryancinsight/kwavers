use crate::gpu::memory::GpuMemoryPoolType;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::fft::{Complex64, Shape1D, FFT_CACHE_1D};
use log::{debug, info, warn};
use ndarray::{Array1, Array3, Array4};
use rayon::prelude::*;
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
    /// - Propagates any [`KwaversError`] returned by called functions.
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
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn submit_rf_data(&mut self, rf_data: Array4<f32>) -> KwaversResult<()> {
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
    pub fn get_processed_frame(&mut self) -> Option<Array3<f32>> {
        let mut buffer = self.output_buffer.lock().unwrap_or_else(|e| e.into_inner());
        buffer.pop_front()
    }
    /// Process pipeline.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
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

    fn process_frame(&mut self, rf_data: &Array4<f32>) -> KwaversResult<Array3<f32>> {
        let beamformed = self.beamform(rf_data)?;
        let envelope = self.envelope_detection(&beamformed)?;
        let compressed = self.log_compression(&envelope)?;
        self.scan_conversion(&compressed)
    }

    fn beamform(&self, rf_data: &Array4<f32>) -> KwaversResult<Array3<f32>> {
        use ndarray::Axis;
        Ok(rf_data.sum_axis(Axis(0)))
    }

    fn envelope_detection(&mut self, beamformed: &Array3<f32>) -> KwaversResult<Array3<f32>> {
        let mut envelope = Array3::zeros(beamformed.dim());

        use ndarray::Axis;

        envelope
            .axis_iter_mut(Axis(0))
            .zip(beamformed.axis_iter(Axis(0)))
            .par_bridge()
            .for_each(|(mut envelope_rx, beamformed_rx)| {
                let len = beamformed_rx.dim().0;
                if len == 0 {
                    return;
                }

                let plan =
                    FFT_CACHE_1D.get_or_create(Shape1D::new(len).expect("non-zero FFT length"));

                HILBERT_SPECTRUM.with(|spectrum_cell| {
                    let mut spectrum = spectrum_cell.borrow_mut();
                    if spectrum.len() != len {
                        *spectrum = Array1::zeros(len);
                    }

                    for k in 0..beamformed_rx.dim().1 {
                        for j in 0..len {
                            spectrum[j] = Complex64::new(beamformed_rx[[j, k]] as f64, 0.0);
                        }

                        plan.forward_complex_inplace(&mut spectrum);

                        if len > 1 {
                            if len % 2 == 0 {
                                for coeff in spectrum.iter_mut().take(len / 2).skip(1) {
                                    *coeff *= 2.0;
                                }
                                for coeff in spectrum.iter_mut().skip(len / 2 + 1) {
                                    *coeff = Complex64::default();
                                }
                            } else {
                                for coeff in spectrum.iter_mut().take(len / 2 + 1).skip(1) {
                                    *coeff *= 2.0;
                                }
                                for coeff in spectrum.iter_mut().skip(len / 2 + 1) {
                                    *coeff = Complex64::default();
                                }
                            }
                        }

                        plan.inverse_complex_inplace(&mut spectrum);

                        let norm = 1.0 / len as f64;
                        for j in 0..len {
                            envelope_rx[[j, k]] = (spectrum[j] * norm).norm() as f32;
                        }
                    }
                });
            });

        Ok(envelope)
    }

    fn log_compression(&self, envelope: &Array3<f32>) -> KwaversResult<Array3<f32>> {
        let mut compressed = Array3::zeros(envelope.dim());
        let dynamic_range_db = 60.0_f64;
        let compression_factor: f32 = (dynamic_range_db / 20.0) as f32;

        for i in 0..envelope.shape()[0] {
            for j in 0..envelope.shape()[1] {
                for k in 0..envelope.dim().2 {
                    let value = envelope[[i, j, k]].max(1e-10) as f64;
                    let log_value = value.ln() / (compression_factor as f64).ln();
                    compressed[[i, j, k]] = log_value.clamp(0.0, 1.0) as f32;
                }
            }
        }

        Ok(compressed)
    }

    fn scan_conversion(&self, compressed: &Array3<f32>) -> KwaversResult<Array3<f32>> {
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
