//! Real-Time Imaging Pipelines for GPU-Accelerated Ultrasound Processing

use crate::gpu::memory::UnifiedMemoryManager;
use kwavers_math::fft::Complex64;
use ndarray::{Array1, Array3, Array4};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

mod realtime;
mod streaming;
#[cfg(test)]
mod tests;

thread_local! {
    pub(super) static HILBERT_SPECTRUM: RefCell<Array1<Complex64>> = RefCell::new(Array1::zeros(0));
}

/// Configuration for real-time imaging pipeline
#[derive(Debug, Clone)]
pub struct RealtimePipelineConfig {
    pub target_fps: f64,
    pub max_latency_ms: f64,
    pub buffer_size: usize,
    pub gpu_accelerated: bool,
    pub adaptive_processing: bool,
    pub streaming_mode: bool,
}

/// Real-time imaging pipeline
#[derive(Debug)]
pub struct RealtimeImagingPipeline {
    pub(super) config: RealtimePipelineConfig,
    pub(super) input_buffer: Arc<Mutex<VecDeque<Array4<f32>>>>,
    pub(super) output_buffer: Arc<Mutex<VecDeque<Array3<f32>>>>,
    pub(super) gpu_memory: Option<UnifiedMemoryManager>,
    pub(super) stats: PipelineStats,
    pub(super) state: PipelineState,
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

/// Streaming data source for real-time imaging
#[derive(Debug)]
pub struct StreamingDataSource {
    pub(super) config: StreamingConfig,
    pub(super) generation_thread: Option<std::thread::JoinHandle<()>>,
    pub(super) stop_signal: Arc<Mutex<bool>>,
}

/// Configuration for streaming data source
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    pub frame_rate: f64,
    pub frame_size: (usize, usize, usize, usize),
    pub noise_level: f64,
    pub signal_amplitude: f64,
    pub source_id: String,
    pub sample_rate: f64,
}
