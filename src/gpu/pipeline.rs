//! Real-Time Imaging Pipelines for GPU-Accelerated Ultrasound Processing
//!
//! Implements streaming pipelines for real-time ultrasound imaging with
//! GPU acceleration, adaptive beamforming, and interactive visualization.

use crate::error::KwaversResult;
use crate::gpu::memory::{MemoryPoolType, UnifiedMemoryManager};
use ndarray::{Array3, Array4};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Configuration for real-time imaging pipeline
#[derive(Debug, Clone)]
pub struct RealtimePipelineConfig {
    /// Target frame rate (Hz)
    pub target_fps: f64,
    /// Maximum latency (ms)
    pub max_latency_ms: f64,
    /// Buffer size (number of frames)
    pub buffer_size: usize,
    /// Enable GPU acceleration
    pub gpu_accelerated: bool,
    /// Adaptive processing based on image quality
    pub adaptive_processing: bool,
    /// Streaming mode (continuous processing)
    pub streaming_mode: bool,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct ComputePipeline {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::PipelineLayout,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct PipelineLayout {
    layout: wgpu::PipelineLayout,
}

/// Real-time imaging pipeline
#[derive(Debug)]
pub struct RealtimeImagingPipeline {
    /// Pipeline configuration
    config: RealtimePipelineConfig,
    /// Input data buffer
    input_buffer: Arc<Mutex<VecDeque<Array4<f32>>>>,
    /// Processed frames buffer
    output_buffer: Arc<Mutex<VecDeque<Array3<f32>>>>,
    /// GPU memory manager
    gpu_memory: Option<UnifiedMemoryManager>,
    /// Processing statistics
    stats: PipelineStats,
    /// Pipeline state
    state: PipelineState,
}

/// Pipeline processing statistics
#[derive(Debug, Default)]
pub struct PipelineStats {
    pub frames_processed: usize,
    pub total_processing_time: Duration,
    pub average_latency: Duration,
    pub dropped_frames: usize,
    pub gpu_memory_usage: usize,
}

/// Pipeline operational state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PipelineState {
    Stopped,
    Starting,
    Running,
    Pausing,
    Paused,
    Stopping,
}

impl RealtimeImagingPipeline {
    /// Create new real-time imaging pipeline
    pub fn new(config: RealtimePipelineConfig) -> KwaversResult<Self> {
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

    /// Start the imaging pipeline
    pub fn start(&mut self) -> KwaversResult<()> {
        println!("Starting real-time imaging pipeline...");
        println!(
            "Target FPS: {:.1}, Max latency: {:.1} ms",
            self.config.target_fps, self.config.max_latency_ms
        );

        self.state = PipelineState::Starting;

        // Initialize GPU memory pools if needed
        if let Some(ref mut gpu_mem) = self.gpu_memory {
            // Allocate memory pools for real-time processing
            gpu_mem.allocate(0, MemoryPoolType::Temporary, 1024 * 1024)?; // 1MB temp buffer
            gpu_mem.allocate(0, MemoryPoolType::Persistent, 512 * 1024)?; // 512KB persistent
        }

        self.state = PipelineState::Running;
        println!("Pipeline started successfully");
        Ok(())
    }

    /// Stop the imaging pipeline
    pub fn stop(&mut self) -> KwaversResult<()> {
        println!("Stopping real-time imaging pipeline...");
        self.state = PipelineState::Stopping;

        // Clear buffers
        {
            let mut input = self.input_buffer.lock().unwrap();
            input.clear();
        }
        {
            let mut output = self.output_buffer.lock().unwrap();
            output.clear();
        }

        self.state = PipelineState::Stopped;
        println!("Pipeline stopped");
        Ok(())
    }

    /// Submit RF data for processing
    pub fn submit_rf_data(&mut self, rf_data: Array4<f32>) -> KwaversResult<()> {
        if self.state != PipelineState::Running {
            return Err(crate::error::KwaversError::InvalidInput(
                "Pipeline is not running".to_string(),
            ));
        }

        let mut buffer = self.input_buffer.lock().unwrap();

        // Check buffer capacity
        if buffer.len() >= self.config.buffer_size {
            self.stats.dropped_frames += 1;
            println!("Warning: Input buffer full, dropping frame");
            return Ok(());
        }

        buffer.push_back(rf_data);
        Ok(())
    }

    /// Retrieve processed image frame
    pub fn get_processed_frame(&mut self) -> Option<Array3<f32>> {
        let mut buffer = self.output_buffer.lock().unwrap();
        buffer.pop_front()
    }

    /// Process pending frames in the pipeline
    pub fn process_pipeline(&mut self) -> KwaversResult<()> {
        if self.state != PipelineState::Running {
            return Ok(());
        }

        let start_time = Instant::now();

        // Get input data
        let rf_data = {
            let mut buffer = self.input_buffer.lock().unwrap();
            buffer.pop_front()
        };

        if let Some(data) = rf_data {
            // Process the frame
            let processed_frame = self.process_frame(&data)?;

            // Store result
            {
                let mut output = self.output_buffer.lock().unwrap();
                if output.len() < self.config.buffer_size {
                    output.push_back(processed_frame);
                } else {
                    self.stats.dropped_frames += 1;
                }
            }

            // Update statistics
            let processing_time = start_time.elapsed();
            self.stats.frames_processed += 1;
            self.stats.total_processing_time += processing_time;

            if self.stats.frames_processed > 0 {
                self.stats.average_latency =
                    self.stats.total_processing_time / self.stats.frames_processed as u32;
            }

            // Check latency requirements
            let latency_ms = processing_time.as_secs_f64() * 1000.0;
            if latency_ms > self.config.max_latency_ms {
                println!(
                    "Warning: Processing latency {:.1}ms exceeds limit {:.1}ms",
                    latency_ms, self.config.max_latency_ms
                );
            }
        }

        Ok(())
    }

    /// Process a single frame through the imaging pipeline
    fn process_frame(&self, rf_data: &Array4<f32>) -> KwaversResult<Array3<f32>> {
        // Step 1: Beamforming
        let beamformed = self.beamform(rf_data)?;

        // Step 2: Envelope detection
        let envelope = self.envelope_detection(&beamformed)?;

        // Step 3: Log compression
        let compressed = self.log_compression(&envelope)?;

        // Step 4: Scan conversion (if needed)
        let scan_converted = self.scan_conversion(&compressed)?;

        Ok(scan_converted)
    }

    /// Beamforming step
    fn beamform(&self, rf_data: &Array4<f32>) -> KwaversResult<Array3<f32>> {
        // Simplified beamforming for real-time processing
        // In practice, this would use the full beamforming pipeline
        let (n_tx, n_rx, n_samples, n_frames) = rf_data.dim();

        // Sum across transmitters for simple delay-and-sum
        let mut beamformed = Array3::zeros((n_rx, n_samples, n_frames));

        for tx in 0..n_tx {
            for rx in 0..n_rx {
                for sample in 0..n_samples {
                    for frame in 0..n_frames {
                        beamformed[[rx, sample, frame]] += rf_data[[tx, rx, sample, frame]];
                    }
                }
            }
        }

        Ok(beamformed)
    }

    /// Envelope detection
    fn envelope_detection(&self, beamformed: &Array3<f32>) -> KwaversResult<Array3<f32>> {
        // Hilbert transform for envelope detection
        let mut envelope = Array3::zeros(beamformed.dim());

        for i in 0..beamformed.shape()[0] {
            for k in 0..beamformed.dim().2 {
                // Simplified envelope detection (magnitude of analytic signal)
                let mut analytic = vec![];

                for j in 0..beamformed.shape()[1] {
                    analytic.push(beamformed[[i, j, k]] as f64);
                }

                // Apply Hilbert transform for analytic signal
                let hilbert = self.hilbert_transform(&analytic);

                for j in 0..hilbert.len() {
                    envelope[[i, j, k]] =
                        ((beamformed[[i, j, k]] as f64).powi(2) + hilbert[j].powi(2)).sqrt() as f32;
                }
            }
        }

        Ok(envelope)
    }

    /// Hilbert transform using FFT-based approach
    /// Computes the analytic signal for envelope detection
    ///
    /// # Theorem Reference
    /// Hilbert Transform: H[f](t) = (1/π) ∫ f(τ)/(t-τ) dτ
    /// FFT-based implementation: Multiply FFT by sign function in frequency domain
    /// Reference: Oppenheim & Schafer, Discrete-Time Signal Processing (3rd ed.)
    fn hilbert_transform(&self, signal: &[f64]) -> Vec<f64> {
        use rustfft::{num_complex::Complex, FftPlanner};

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(signal.len());
        let ifft = planner.plan_fft_inverse(signal.len());

        // Convert to complex
        let mut spectrum: Vec<Complex<f64>> =
            signal.iter().map(|&x| Complex::new(x, 0.0)).collect();

        // Forward FFT
        fft.process(&mut spectrum);

        // Apply Hilbert transform in frequency domain
        let n = spectrum.len();
        for i in 1..n / 2 {
            // Negative frequencies get multiplied by 2, positive by 0
            spectrum[i] *= 2.0;
        }
        // DC and Nyquist remain unchanged
        if n % 2 == 1 {
            // For odd length, Nyquist frequency (if exists) remains unchanged
        } else {
            // For even length, Nyquist frequency remains unchanged
        }

        // Inverse FFT
        ifft.process(&mut spectrum);

        // Return imaginary part (Hilbert transform result)
        spectrum.iter().map(|c| c.im / n as f64).collect()
    }

    /// Log compression
    fn log_compression(&self, envelope: &Array3<f32>) -> KwaversResult<Array3<f32>> {
        let mut compressed = Array3::zeros(envelope.dim());
        let dynamic_range_db = 60.0; // 60 dB dynamic range
        let compression_factor: f32 = dynamic_range_db / 20.0; // Convert to linear factor

        for i in 0..envelope.shape()[0] {
            for j in 0..envelope.shape()[1] {
                for k in 0..envelope.dim().2 {
                    let value = envelope[[i, j, k]].max(1e-10) as f64; // Avoid log(0)
                    let log_value = value.ln() / (compression_factor as f64).ln();
                    compressed[[i, j, k]] = log_value.max(0.0).min(1.0) as f32;
                }
            }
        }

        Ok(compressed)
    }

    /// Scan conversion
    fn scan_conversion(&self, compressed: &Array3<f32>) -> KwaversResult<Array3<f32>> {
        // For now, return as-is (would implement proper scan conversion for curved arrays)
        Ok(compressed.clone())
    }

    /// Get pipeline statistics
    pub fn statistics(&self) -> &PipelineStats {
        &self.stats
    }

    /// Get current pipeline state
    pub fn state(&self) -> PipelineState {
        self.state
    }

    /// Check if pipeline is ready for new data
    pub fn is_ready(&self) -> bool {
        let buffer = self.input_buffer.lock().unwrap();
        buffer.len() < self.config.buffer_size
    }

    /// Get current buffer utilization
    pub fn buffer_utilization(&self) -> (f64, f64) {
        let input_len = self.input_buffer.lock().unwrap().len();
        let output_len = self.output_buffer.lock().unwrap().len();

        (
            input_len as f64 / self.config.buffer_size as f64,
            output_len as f64 / self.config.buffer_size as f64,
        )
    }

    /// Enable/disable adaptive processing
    pub fn set_adaptive_processing(&mut self, enabled: bool) {
        self.config.adaptive_processing = enabled;
    }

    /// Adjust target frame rate
    pub fn set_target_fps(&mut self, fps: f64) {
        self.config.target_fps = fps;
    }
}

/// Streaming data source for real-time imaging
#[derive(Debug)]
pub struct StreamingDataSource {
    /// Configuration
    config: StreamingConfig,
    /// Data generation thread handle
    generation_thread: Option<std::thread::JoinHandle<()>>,
    /// Stop signal
    stop_signal: Arc<Mutex<bool>>,
}

/// Configuration for streaming data source
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    pub frame_rate: f64,
    pub frame_size: (usize, usize, usize, usize), // (tx, rx, samples, frames)
    pub noise_level: f64,
    pub signal_amplitude: f64,
    pub source_id: String,
    pub sample_rate: f64,
}

impl StreamingDataSource {
    /// Create new streaming data source
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            config,
            generation_thread: None,
            stop_signal: Arc::new(Mutex::new(false)),
        }
    }

    /// Start streaming data
    pub fn start_streaming(
        &mut self,
        pipeline: Arc<Mutex<RealtimeImagingPipeline>>,
    ) -> KwaversResult<()> {
        let config = self.config.clone();
        let stop_signal = Arc::clone(&self.stop_signal);

        let handle = std::thread::spawn(move || {
            let frame_interval = Duration::from_secs_f64(1.0 / config.frame_rate);

            loop {
                let start_time = Instant::now();

                // Check stop signal
                {
                    let stop = stop_signal.lock().unwrap();
                    if *stop {
                        break;
                    }
                }

                // Generate synthetic RF data
                let rf_data = Self::generate_rf_frame(&config);

                // Submit to pipeline
                {
                    let mut pipeline_lock = pipeline.lock().unwrap();
                    let _ = pipeline_lock.submit_rf_data(rf_data);
                }

                // Wait for next frame
                let elapsed = start_time.elapsed();
                if elapsed < frame_interval {
                    std::thread::sleep(frame_interval - elapsed);
                }
            }
        });

        self.generation_thread = Some(handle);
        Ok(())
    }

    /// Stop streaming data
    pub fn stop_streaming(&mut self) -> KwaversResult<()> {
        {
            let mut stop = self.stop_signal.lock().unwrap();
            *stop = true;
        }

        if let Some(handle) = self.generation_thread.take() {
            let _ = handle.join();
        }

        Ok(())
    }

    /// Generate a single RF frame
    fn generate_rf_frame(config: &StreamingConfig) -> Array4<f32> {
        let (n_tx, n_rx, n_samples, n_frames) = config.frame_size;
        let mut rf_data = Array4::zeros((n_tx, n_rx, n_samples, n_frames));

        // Generate realistic ultrasound RF signals
        for tx in 0..n_tx {
            for rx in 0..n_rx {
                for sample in 0..n_samples {
                    for frame in 0..n_frames {
                        let time = sample as f64 * 1e-8; // 10 ns sampling
                        let delay = (tx + rx) as f64 * 1e-7; // Inter-element delay

                        // Generate ultrasound pulse with noise
                        let signal = config.signal_amplitude
                            * (-0.5 * ((time - delay) * 5e7).powi(2)).exp()
                            * ((time - delay) * 3e7).cos();

                        let noise = config.noise_level * rand::random::<f64>();
                        rf_data[[tx, rx, sample, frame]] = (signal + noise) as f32;
                    }
                }
            }
        }

        rf_data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let config = RealtimePipelineConfig {
            target_fps: 30.0,
            max_latency_ms: 33.0,
            buffer_size: 10,
            gpu_accelerated: false,
            adaptive_processing: true,
            streaming_mode: false,
        };

        let pipeline = RealtimeImagingPipeline::new(config);
        assert!(pipeline.is_ok());
    }

    #[test]
    fn test_pipeline_start_stop() {
        let config = RealtimePipelineConfig {
            target_fps: 10.0,
            max_latency_ms: 100.0,
            buffer_size: 5,
            gpu_accelerated: false,
            adaptive_processing: false,
            streaming_mode: false,
        };

        let mut pipeline = RealtimeImagingPipeline::new(config).unwrap();

        assert!(pipeline.start().is_ok());
        assert_eq!(pipeline.state(), PipelineState::Running);

        assert!(pipeline.stop().is_ok());
        assert_eq!(pipeline.state(), PipelineState::Stopped);
    }

    #[test]
    fn test_data_submission() {
        let config = RealtimePipelineConfig {
            target_fps: 10.0,
            max_latency_ms: 100.0,
            buffer_size: 5,
            gpu_accelerated: false,
            adaptive_processing: false,
            streaming_mode: false,
        };

        let mut pipeline = RealtimeImagingPipeline::new(config).unwrap();
        pipeline.start().unwrap();

        // Submit test data
        let rf_data = Array4::from_elem((4, 32, 1024, 1), 1.0);
        assert!(pipeline.submit_rf_data(rf_data).is_ok());

        pipeline.stop().unwrap();
    }

    #[test]
    fn test_frame_processing() {
        let config = RealtimePipelineConfig {
            target_fps: 10.0,
            max_latency_ms: 100.0,
            buffer_size: 5,
            gpu_accelerated: false,
            adaptive_processing: false,
            streaming_mode: false,
        };

        let mut pipeline = RealtimeImagingPipeline::new(config).unwrap();
        pipeline.start().unwrap();

        // Submit and process data
        let rf_data = Array4::from_elem((2, 16, 512, 1), 1.0);
        pipeline.submit_rf_data(rf_data).unwrap();
        pipeline.process_pipeline().unwrap();

        // Check that processed frame is available
        let processed = pipeline.get_processed_frame();
        assert!(processed.is_some());

        let frame = processed.unwrap();
        assert_eq!(frame.dim(), (16, 512, 1));

        pipeline.stop().unwrap();
    }

    #[test]
    fn test_streaming_data_source() {
        // Configure high frame rate for quick test and small buffers
        let stream_cfg = StreamingConfig {
            frame_rate: 50.0,
            frame_size: (2, 8, 128, 1),
            noise_level: 0.05,
            signal_amplitude: 1.0,
            source_id: "test".to_string(),
            sample_rate: 40_000_000.0,
        };

        let mut data_source = StreamingDataSource::new(stream_cfg);

        // Prepare a minimal pipeline and start streaming briefly
        let mut pipeline = RealtimeImagingPipeline::new(RealtimePipelineConfig {
            target_fps: 60.0,
            max_latency_ms: 50.0,
            buffer_size: 8,
            gpu_accelerated: false,
            adaptive_processing: false,
            streaming_mode: true,
        })
        .unwrap();

        pipeline.start().unwrap();
        let pipeline_arc = std::sync::Arc::new(std::sync::Mutex::new(pipeline));
        data_source
            .start_streaming(std::sync::Arc::clone(&pipeline_arc))
            .unwrap();

        // Allow a few frames to be generated
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Stop streaming and inspect buffer utilization
        data_source.stop_streaming().unwrap();
        let mut pipeline_lock = pipeline_arc.lock().unwrap();
        let (input_util, _output_util) = pipeline_lock.buffer_utilization();
        assert!(input_util > 0.0);

        pipeline_lock.stop().unwrap();
    }
}
