//! Bounded visualization frame streaming.
//!
//! The stream surface is CPU-side orchestration for GPU visualization. It uses
//! bounded channels and Leto frame buffers so visualization does not introduce
//! new ndarray-facing APIs.

use kwavers_grid::Grid;
use leto::Array3 as LetoArray3;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const DEFAULT_CHANNEL_CAPACITY: usize = 64;

/// Monotonic visualization frame identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FrameId(pub u64);

impl FrameId {
    /// Return the next unique frame identifier from `counter`.
    pub fn next(counter: &AtomicU64) -> Self {
        Self(counter.fetch_add(1, Ordering::Relaxed))
    }
}

/// Per-frame simulation metadata.
#[derive(Debug, Clone)]
pub struct FrameMetadata {
    /// Monotonic frame identifier.
    pub id: FrameId,
    /// Simulation time represented by the frame, in seconds.
    pub simulation_time: f64,
    /// Grid associated with the frame.
    pub grid: Grid,
    /// Quality scaling factor used when producing the frame.
    pub quality_factor: f64,
    /// Caller-defined labels.
    pub tags: Vec<String>,
}

/// Visualization frame payload.
#[derive(Debug, Clone)]
pub struct VizFrame {
    /// Pressure field to visualize.
    pub pressure: LetoArray3<f32>,
    /// Optional uncertainty or auxiliary scalar field.
    pub auxiliary: Option<LetoArray3<f32>>,
    /// Frame metadata.
    pub metadata: FrameMetadata,
    created_at: Instant,
}

impl VizFrame {
    /// Construct a visualization frame.
    pub fn new(
        pressure: LetoArray3<f32>,
        auxiliary: Option<LetoArray3<f32>>,
        metadata: FrameMetadata,
    ) -> Self {
        Self {
            pressure,
            auxiliary,
            metadata,
            created_at: Instant::now(),
        }
    }

    /// Return elapsed wall-clock age since frame construction.
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

/// Bounded buffering policy for visualization frames.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BufferPolicy {
    /// Block the producer when the bounded channel is full.
    Block,
    /// Drop the newest frame when the bounded channel is full.
    DropLatest,
    /// Drop the oldest buffered frame to make room.
    DropOldest(usize),
    /// Drop newest frames once measured latency exceeds this target.
    AdaptiveLatency(f64),
}

impl BufferPolicy {
    fn capacity(self) -> usize {
        match self {
            Self::DropOldest(capacity) => capacity.max(1),
            Self::Block | Self::DropLatest | Self::AdaptiveLatency(_) => DEFAULT_CHANNEL_CAPACITY,
        }
    }
}

/// Streaming counters.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StreamStatistics {
    /// Frames accepted by the stream.
    pub frames_produced: u64,
    /// Frames consumed by receivers.
    pub frames_consumed: u64,
    /// Frames dropped by buffering policy.
    pub frames_dropped: u64,
    /// Exponential moving average of frame age at receive time.
    pub avg_latency_ms: f64,
}

#[derive(Debug)]
struct VizStreamInner {
    policy: BufferPolicy,
    tx: flume::Sender<VizFrame>,
    rx: flume::Receiver<VizFrame>,
    stats: Mutex<StreamStatistics>,
    closed: AtomicBool,
    pool: Option<Arc<FramePool>>,
}

impl VizStreamInner {
    async fn send_frame(&self, frame: VizFrame) -> Result<(), String> {
        if self.closed.load(Ordering::Acquire) {
            return Err("visualization stream is closed".to_string());
        }

        match self.policy {
            BufferPolicy::Block => self
                .tx
                .send_async(frame)
                .await
                .map_err(|err| format!("send failed: {err}"))
                .map(|()| self.record_produced()),
            BufferPolicy::DropLatest => match self.tx.try_send(frame) {
                Ok(()) => {
                    self.record_produced();
                    Ok(())
                }
                Err(flume::TrySendError::Full(_)) => {
                    self.record_dropped();
                    Ok(())
                }
                Err(flume::TrySendError::Disconnected(_)) => {
                    Err("visualization stream is disconnected".to_string())
                }
            },
            BufferPolicy::DropOldest(_) => self.send_drop_oldest(frame),
            BufferPolicy::AdaptiveLatency(target_ms) => {
                if self.statistics().avg_latency_ms > target_ms && self.tx.is_full() {
                    self.record_dropped();
                    Ok(())
                } else {
                    self.tx
                        .send_async(frame)
                        .await
                        .map_err(|err| format!("send failed: {err}"))
                        .map(|()| self.record_produced())
                }
            }
        }
    }

    fn send_frame_blocking(&self, frame: VizFrame) -> Result<(), String> {
        if self.closed.load(Ordering::Acquire) {
            return Err("visualization stream is closed".to_string());
        }

        match self.policy {
            BufferPolicy::Block => self
                .tx
                .send(frame)
                .map_err(|err| format!("send failed: {err}"))
                .map(|()| self.record_produced()),
            BufferPolicy::DropLatest => match self.tx.try_send(frame) {
                Ok(()) => {
                    self.record_produced();
                    Ok(())
                }
                Err(flume::TrySendError::Full(_)) => {
                    self.record_dropped();
                    Ok(())
                }
                Err(flume::TrySendError::Disconnected(_)) => {
                    Err("visualization stream is disconnected".to_string())
                }
            },
            BufferPolicy::DropOldest(_) => self.send_drop_oldest(frame),
            BufferPolicy::AdaptiveLatency(target_ms) => {
                if self.statistics().avg_latency_ms > target_ms && self.tx.is_full() {
                    self.record_dropped();
                    Ok(())
                } else {
                    self.tx
                        .send(frame)
                        .map_err(|err| format!("send failed: {err}"))
                        .map(|()| self.record_produced())
                }
            }
        }
    }

    fn send_drop_oldest(&self, mut frame: VizFrame) -> Result<(), String> {
        loop {
            match self.tx.try_send(frame) {
                Ok(()) => {
                    self.record_produced();
                    return Ok(());
                }
                Err(flume::TrySendError::Full(returned)) => {
                    frame = returned;
                    match self.rx.try_recv() {
                        Ok(_) => self.record_dropped(),
                        Err(flume::TryRecvError::Empty) => continue,
                        Err(flume::TryRecvError::Disconnected) => {
                            return Err("visualization stream is disconnected".to_string());
                        }
                    }
                }
                Err(flume::TrySendError::Disconnected(_)) => {
                    return Err("visualization stream is disconnected".to_string());
                }
            }
        }
    }

    async fn recv_frame(&self) -> Result<VizFrame, String> {
        self.rx
            .recv_async()
            .await
            .map_err(|err| format!("receive failed: {err}"))
            .map(|frame| {
                self.record_consumed(frame.age());
                frame
            })
    }

    fn recv_frame_blocking(&self) -> Result<VizFrame, String> {
        self.rx
            .recv()
            .map_err(|err| format!("receive failed: {err}"))
            .map(|frame| {
                self.record_consumed(frame.age());
                frame
            })
    }

    fn try_recv_frame(&self) -> Result<Option<VizFrame>, String> {
        match self.rx.try_recv() {
            Ok(frame) => {
                self.record_consumed(frame.age());
                Ok(Some(frame))
            }
            Err(flume::TryRecvError::Empty) => Ok(None),
            Err(flume::TryRecvError::Disconnected) => {
                Err("visualization stream is disconnected".to_string())
            }
        }
    }

    fn record_produced(&self) {
        let mut stats = self.stats.lock().expect("invariant: stream stats lock");
        stats.frames_produced += 1;
    }

    fn record_consumed(&self, latency: Duration) {
        let mut stats = self.stats.lock().expect("invariant: stream stats lock");
        stats.frames_consumed += 1;
        update_ema(&mut stats.avg_latency_ms, latency.as_secs_f64() * 1_000.0);
    }

    fn record_dropped(&self) {
        let mut stats = self.stats.lock().expect("invariant: stream stats lock");
        stats.frames_dropped += 1;
    }

    fn statistics(&self) -> StreamStatistics {
        self.stats
            .lock()
            .expect("invariant: stream stats lock")
            .clone()
    }
}

/// Visualization stream with bounded producer/consumer queues.
#[derive(Debug, Clone)]
pub struct VizStream {
    inner: Arc<VizStreamInner>,
}

impl VizStream {
    /// Construct a stream with a bounded channel selected by `policy`.
    pub fn new(policy: BufferPolicy) -> Result<Self, String> {
        let (tx, rx) = flume::bounded(policy.capacity());
        Ok(Self {
            inner: Arc::new(VizStreamInner {
                policy,
                tx,
                rx,
                stats: Mutex::new(StreamStatistics::default()),
                closed: AtomicBool::new(false),
                pool: None,
            }),
        })
    }

    /// Construct a stream with a reusable frame pool.
    pub fn with_pool(
        policy: BufferPolicy,
        dimensions: (usize, usize, usize),
        pool_size: usize,
    ) -> Result<Self, String> {
        let stream = Self::new(policy)?;
        let inner = VizStreamInner {
            policy,
            tx: stream.inner.tx.clone(),
            rx: stream.inner.rx.clone(),
            stats: Mutex::new(StreamStatistics::default()),
            closed: AtomicBool::new(false),
            pool: Some(Arc::new(FramePool::new(
                dimensions.0,
                dimensions.1,
                dimensions.2,
                pool_size,
            ))),
        };
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Return the configured buffering policy.
    pub fn policy(&self) -> BufferPolicy {
        self.inner.policy
    }

    /// Return a cloneable frame sender.
    pub fn sender(&self) -> StreamSender {
        StreamSender {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Return a cloneable frame receiver.
    pub fn receiver(&self) -> StreamReceiver {
        StreamReceiver {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Send a frame through the stream.
    pub async fn send_frame(&self, frame: VizFrame) -> Result<(), String> {
        self.inner.send_frame(frame).await
    }

    /// Send a frame through the stream without requiring an async runtime.
    pub fn send_frame_blocking(&self, frame: VizFrame) -> Result<(), String> {
        self.inner.send_frame_blocking(frame)
    }

    /// Receive the next frame.
    pub async fn recv_frame(&self) -> Result<VizFrame, String> {
        self.inner.recv_frame().await
    }

    /// Receive the next frame without requiring an async runtime.
    pub fn recv_frame_blocking(&self) -> Result<VizFrame, String> {
        self.inner.recv_frame_blocking()
    }

    /// Try to receive a frame without blocking.
    pub fn try_recv_frame(&self) -> Result<Option<VizFrame>, String> {
        self.inner.try_recv_frame()
    }

    /// Return a snapshot of stream statistics.
    pub fn statistics(&self) -> StreamStatistics {
        self.inner.statistics()
    }

    /// Return true when a frame pool is attached.
    pub fn has_pool(&self) -> bool {
        self.inner.pool.is_some()
    }

    /// Borrow the shared frame pool, if present.
    pub fn frame_pool(&self) -> Option<Arc<FramePool>> {
        self.inner.pool.as_ref().map(Arc::clone)
    }

    /// Estimate bounded pool memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.inner
            .pool
            .as_ref()
            .map_or(0, |pool| pool.memory_usage_bytes())
    }

    /// Close the stream for future sends.
    pub fn close(&self) {
        self.inner.closed.store(true, Ordering::Release);
    }

    /// Return true after `close` is called.
    pub fn is_closed(&self) -> bool {
        self.inner.closed.load(Ordering::Acquire)
    }
}

/// Cloneable stream sender.
#[derive(Debug, Clone)]
pub struct StreamSender {
    inner: Arc<VizStreamInner>,
}

impl StreamSender {
    /// Send a frame asynchronously.
    pub async fn send_async(&self, frame: VizFrame) -> Result<(), String> {
        self.inner.send_frame(frame).await
    }

    /// Send a frame without requiring an async runtime.
    pub fn send_blocking(&self, frame: VizFrame) -> Result<(), String> {
        self.inner.send_frame_blocking(frame)
    }
}

/// Cloneable stream receiver.
#[derive(Debug, Clone)]
pub struct StreamReceiver {
    inner: Arc<VizStreamInner>,
}

impl StreamReceiver {
    /// Receive a frame asynchronously.
    pub async fn recv_async(&self) -> Result<VizFrame, String> {
        self.inner.recv_frame().await
    }

    /// Receive a frame without requiring an async runtime.
    pub fn recv_blocking(&self) -> Result<VizFrame, String> {
        self.inner.recv_frame_blocking()
    }
}

/// Reusable pool of Leto frame buffers.
#[derive(Debug)]
pub struct FramePool {
    dimensions: [usize; 3],
    capacity: usize,
    buffers: Mutex<Vec<LetoArray3<f32>>>,
}

impl FramePool {
    /// Construct a frame pool.
    pub fn new(nx: usize, ny: usize, nz: usize, capacity: usize) -> Self {
        let dimensions = [nx, ny, nz];
        let buffers = (0..capacity)
            .map(|_| LetoArray3::<f32>::zeros(dimensions))
            .collect();
        Self {
            dimensions,
            capacity,
            buffers: Mutex::new(buffers),
        }
    }

    /// Acquire a frame buffer.
    pub fn acquire(&self) -> LetoArray3<f32> {
        self.buffers
            .lock()
            .expect("invariant: frame pool lock")
            .pop()
            .unwrap_or_else(|| LetoArray3::<f32>::zeros(self.dimensions))
    }

    /// Return a frame buffer to the pool.
    pub fn release(&self, buffer: LetoArray3<f32>) {
        let mut buffers = self.buffers.lock().expect("invariant: frame pool lock");
        if buffers.len() < self.capacity && buffer.shape() == self.dimensions {
            buffers.push(buffer);
        }
    }

    fn memory_usage_bytes(&self) -> usize {
        self.capacity * self.dimensions.iter().product::<usize>() * std::mem::size_of::<f32>()
    }
}

/// Synchronization statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct SyncStatistics {
    /// Rendered frame count.
    pub frames_rendered: u64,
    /// Dropped frame count.
    pub frames_dropped: u64,
    /// Current average latency in milliseconds.
    pub latency_ms: f64,
    /// Target frame rate.
    pub target_fps: f64,
    /// Current quality factor.
    pub quality_level: f64,
    /// Percentage of total frames dropped.
    pub drop_rate_percent: f64,
}

impl Default for SyncStatistics {
    fn default() -> Self {
        Self {
            frames_rendered: 0,
            frames_dropped: 0,
            latency_ms: 0.0,
            target_fps: 0.0,
            quality_level: 1.0,
            drop_rate_percent: 0.0,
        }
    }
}

#[derive(Debug)]
struct SyncState {
    stats: SyncStatistics,
    quality: QualityLevel,
}

/// Coordinates frame pacing and adaptive quality.
#[derive(Debug, Clone)]
pub struct SyncCoordinator {
    target_fps: f64,
    paused: Arc<AtomicBool>,
    state: Arc<Mutex<SyncState>>,
}

impl SyncCoordinator {
    /// Construct a synchronization coordinator.
    pub fn new(target_fps: f64) -> Self {
        Self {
            target_fps,
            paused: Arc::new(AtomicBool::new(false)),
            state: Arc::new(Mutex::new(SyncState {
                stats: SyncStatistics {
                    target_fps,
                    ..SyncStatistics::default()
                },
                quality: QualityLevel::Maximum,
            })),
        }
    }

    /// Pause frame presentation.
    pub fn pause(&self) {
        self.paused.store(true, Ordering::Release);
    }

    /// Resume frame presentation.
    pub fn resume(&self) {
        self.paused.store(false, Ordering::Release);
    }

    /// Return true when paused.
    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Acquire)
    }

    /// Return the current quality factor.
    pub fn quality_factor(&self) -> f64 {
        self.state
            .lock()
            .expect("invariant: sync state lock")
            .quality
            .factor()
    }

    /// Set quality level manually.
    pub fn set_quality(&self, quality: QualityLevel) {
        let mut state = self.state.lock().expect("invariant: sync state lock");
        state.quality = quality;
        state.stats.quality_level = quality.factor();
    }

    /// Record one completed frame.
    pub fn complete_frame(&self, latency: Duration, _simulation_time: f64) {
        let mut state = self.state.lock().expect("invariant: sync state lock");
        state.stats.frames_rendered += 1;
        update_ema(&mut state.stats.latency_ms, latency.as_secs_f64() * 1_000.0);

        let budget_ms = 1_000.0 / self.target_fps.max(f64::EPSILON);
        if state.stats.latency_ms > budget_ms * 1.15 {
            state.quality = state.quality.downgrade();
        } else if state.stats.latency_ms < budget_ms * 0.65 {
            state.quality = state.quality.upgrade();
        }
        state.stats.quality_level = state.quality.factor();
        refresh_drop_rate(&mut state.stats);
    }

    /// Record a dropped frame.
    pub fn report_drop(&self, _reason: &str) {
        let mut state = self.state.lock().expect("invariant: sync state lock");
        state.stats.frames_dropped += 1;
        refresh_drop_rate(&mut state.stats);
    }

    /// Return current synchronization statistics.
    pub fn statistics(&self) -> SyncStatistics {
        self.state
            .lock()
            .expect("invariant: sync state lock")
            .stats
            .clone()
    }
}

/// Stage-pipeline configuration.
pub mod pipeline {
    /// Visualization pipeline timing and buffering configuration.
    #[derive(Debug, Clone, PartialEq)]
    pub struct PipelineConfig {
        /// Target presentation frame rate.
        pub target_fps: f64,
        /// Bounded input channel capacity.
        pub channel_capacity: usize,
        /// Whether independent stages may execute in parallel.
        pub parallel_execution: bool,
        /// Whether quality adapts to measured latency.
        pub adaptive_quality: bool,
        /// Latency threshold in milliseconds.
        pub latency_threshold_ms: f64,
    }

    impl PipelineConfig {
        /// Return the frame budget in milliseconds.
        pub fn frame_budget_ms(&self) -> f64 {
            1_000.0 / self.target_fps.max(f64::EPSILON)
        }

        /// Return the per-stage budget in milliseconds.
        pub fn stage_budget_ms(&self, stages: usize) -> f64 {
            self.frame_budget_ms() / stages.max(1) as f64
        }
    }

    impl Default for PipelineConfig {
        fn default() -> Self {
            Self {
                target_fps: 30.0,
                channel_capacity: 16,
                parallel_execution: false,
                adaptive_quality: true,
                latency_threshold_ms: 35.0,
            }
        }
    }
}

/// Synchronization quality controls.
pub mod sync {
    use super::SyncStatistics;

    /// Adaptive visualization quality level.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub enum QualityLevel {
        /// Minimal quality.
        Minimal,
        /// Low quality.
        Low,
        /// Medium quality.
        Medium,
        /// High quality.
        High,
        /// Maximum quality.
        Maximum,
    }

    impl QualityLevel {
        /// Return one lower quality level, clamped at `Minimal`.
        pub fn downgrade(self) -> Self {
            match self {
                Self::Maximum => Self::High,
                Self::High => Self::Medium,
                Self::Medium => Self::Low,
                Self::Low | Self::Minimal => Self::Minimal,
            }
        }

        /// Return one higher quality level, clamped at `Maximum`.
        pub fn upgrade(self) -> Self {
            match self {
                Self::Minimal => Self::Low,
                Self::Low => Self::Medium,
                Self::Medium => Self::High,
                Self::High | Self::Maximum => Self::Maximum,
            }
        }

        /// Return the scalar quality factor.
        pub fn factor(self) -> f64 {
            match self {
                Self::Minimal => 0.1,
                Self::Low => 0.3,
                Self::Medium => 0.5,
                Self::High => 0.8,
                Self::Maximum => 1.0,
            }
        }
    }

    /// Emit synchronization metrics through the logging facade.
    pub fn report_sync_metrics(stats: &SyncStatistics) {
        log::info!(
            "visualization sync: rendered={} dropped={} latency_ms={:.3} drop_rate={:.3}%",
            stats.frames_rendered,
            stats.frames_dropped,
            stats.latency_ms,
            stats.drop_rate_percent
        );
    }
}

pub use pipeline::PipelineConfig;
pub use sync::{report_sync_metrics, QualityLevel};

/// Per-stage metrics for the asynchronous visualization pipeline.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PipelineStageMetrics {
    /// Stage name.
    pub name: String,
    /// Processed frame count.
    pub frames_processed: u64,
}

/// Pipeline metrics snapshot.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PipelineMetrics {
    /// Stage-level metrics.
    pub stages: Vec<PipelineStageMetrics>,
    /// Total end-to-end latency in milliseconds.
    pub total_latency_ms: f64,
    /// Dropped-frame percentage.
    pub drop_rate_percent: f64,
}

/// Bounded asynchronous visualization pipeline.
#[derive(Debug, Clone)]
pub struct StagePipeline {
    metrics: Arc<Mutex<PipelineMetrics>>,
    _worker: Arc<std::thread::JoinHandle<()>>,
}

impl StagePipeline {
    /// Construct the stage pipeline and its input sender.
    pub async fn new(config: PipelineConfig) -> Result<(Self, PipelineInputSender), String> {
        Self::new_blocking(config)
    }

    /// Construct the stage pipeline and its input sender without requiring an
    /// async runtime.
    pub fn new_blocking(config: PipelineConfig) -> Result<(Self, PipelineInputSender), String> {
        let (tx, rx) = flume::bounded::<VizFrame>(config.channel_capacity.max(1));
        let metrics = Arc::new(Mutex::new(PipelineMetrics {
            stages: vec![
                PipelineStageMetrics {
                    name: "extract".to_string(),
                    frames_processed: 0,
                },
                PipelineStageMetrics {
                    name: "render".to_string(),
                    frames_processed: 0,
                },
                PipelineStageMetrics {
                    name: "encode".to_string(),
                    frames_processed: 0,
                },
            ],
            total_latency_ms: 0.0,
            drop_rate_percent: 0.0,
        }));
        let worker_metrics = Arc::clone(&metrics);
        let worker = std::thread::spawn(move || {
            while let Ok(frame) = rx.recv() {
                let latency_ms = frame.age().as_secs_f64() * 1_000.0;
                let mut metrics = worker_metrics
                    .lock()
                    .expect("invariant: pipeline metrics lock");
                for stage in &mut metrics.stages {
                    stage.frames_processed += 1;
                }
                update_ema(&mut metrics.total_latency_ms, latency_ms);
            }
        });

        Ok((
            Self {
                metrics,
                _worker: Arc::new(worker),
            },
            PipelineInputSender { tx },
        ))
    }

    /// Return pipeline metrics.
    pub fn metrics(&self) -> PipelineMetrics {
        self.metrics
            .lock()
            .expect("invariant: pipeline metrics lock")
            .clone()
    }
}

/// Sender for the visualization stage pipeline.
#[derive(Debug, Clone)]
pub struct PipelineInputSender {
    tx: flume::Sender<VizFrame>,
}

impl PipelineInputSender {
    /// Send a frame to the pipeline.
    pub async fn send(&self, frame: VizFrame) -> Result<(), String> {
        self.tx
            .send_async(frame)
            .await
            .map_err(|err| format!("pipeline send failed: {err}"))
    }

    /// Send a frame to the pipeline without requiring an async runtime.
    pub fn send_blocking(&self, frame: VizFrame) -> Result<(), String> {
        self.tx
            .send(frame)
            .map_err(|err| format!("pipeline send failed: {err}"))
    }
}

fn update_ema(current: &mut f64, sample: f64) {
    if *current == 0.0 {
        *current = sample;
    } else {
        *current = *current * 0.8 + sample * 0.2;
    }
}

fn refresh_drop_rate(stats: &mut SyncStatistics) {
    let total = stats.frames_rendered + stats.frames_dropped;
    stats.drop_rate_percent = if total == 0 {
        0.0
    } else {
        stats.frames_dropped as f64 / total as f64 * 100.0
    };
}
